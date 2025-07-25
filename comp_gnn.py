import torch
import torch.nn as nn
import torch.nn.functional as F

from roost_core import BaseModelClass
from roost_segments import ResidualNetwork, SimpleNetwork, WeightedAttentionPooling
import torch.nn.init as init



class DescriptorNetwork(nn.Module):
    """
    The Descriptor Network is the message passing section of the
    Roost Model.
    """

    def __init__(
        self,
        elem_emb_len,
        elem_fea_len=64,
        edge_fea_len=2,
        n_graph=3,
        elem_heads=3,
        elem_gate=[256],
        elem_msg=[256],
        cry_heads=3,
        cry_gate=[256],
        cry_msg=[256],
    ):
        """
        """
        super().__init__()

        # apply linear transform to the input to get a trainable embedding
        # NOTE -1 here so we can add the weights as a node feature
        self.embedding = nn.Linear(elem_emb_len, elem_fea_len - 1)
        init.xavier_uniform_(self.embedding.weight)
        init.constant_(self.embedding.bias, 0)
        
        # create a list of Message passing layers
        self.graphs = nn.ModuleList(
            [
                MessageLayer(
                    elem_fea_len=elem_fea_len,
                    edge_fea_len=edge_fea_len,
                    elem_heads=elem_heads,
                    elem_gate=elem_gate,
                    elem_msg=elem_msg,
                )
                for i in range(n_graph)
            ]
        )
        self.bns = nn.ModuleList(
            [
                nn.BatchNorm1d(elem_fea_len) for i in range(n_graph)
            ]
        )

        # define a global pooling function for materials
        self.cry_pool = nn.ModuleList(
            [
                WeightedAttentionPooling(
                    gate_nn=SimpleNetwork(elem_fea_len, 1, cry_gate),
                    message_nn=SimpleNetwork(elem_fea_len, elem_fea_len, cry_msg),
                )
                for _ in range(cry_heads)
            ]
        )

    def forward(self, elem_weights, elem_fea, edge_fea, self_fea_idx, nbr_fea_idx, cry_elem_idx):
        """
        Forward pass

        Parameters
        ----------
        N: Total number of elements (nodes) in the batch
        M: Total number of pairs (edges) in the batch
        C: Total number of crystals (graphs) in the batch

        Inputs
        ----------
        elem_weights: Variable(torch.Tensor) shape (N)
            Fractional weight of each Element in its stoichiometry
        elem_fea: Variable(torch.Tensor) shape (N, orig_elem_fea_len)
            Element features of each of the N elems in the batch
        self_fea_idx: torch.Tensor shape (M,)
            Indices of the first element in each of the M pairs
        nbr_fea_idx: torch.Tensor shape (M,)
            Indices of the second element in each of the M pairs
        cry_elem_idx: list of torch.LongTensor of length C
            Mapping from the elem idx to crystal idx

        Returns
        -------
        cry_fea: nn.Variable shape (C,)
            Material representation after message passing
        """

        # embed the original features into a trainable embedding space
        elem_fea = self.embedding(elem_fea)
        
        # add weights as a node feature
        elem_fea = torch.cat([elem_fea, elem_weights], dim=1)

        # apply the message passing functions
        # for graph_func in self.graphs:
        #     elem_fea = graph_func(elem_weights, elem_fea, self_fea_idx, nbr_fea_idx)
        
        for graph_func, bn in zip(self.graphs, self.bns):
            elem_fea = bn(graph_func(elem_weights, elem_fea, edge_fea, self_fea_idx, nbr_fea_idx)[0])

        # generate crystal features by pooling the elemental features
        head_fea = []
        gate_list = []
        for attnhead in self.cry_pool:
            out,gate = attnhead(elem_fea, index=cry_elem_idx, weights=elem_weights)
            # print(out.shape)
            head_fea.append(out)
            gate_list.append(gate)

        # head_fea = [
        #     head(elem_fea, index=cry_elem_idx, weights=elem_weights)
        #     for head in self.cry_pool
        # ]

        return torch.mean(torch.stack(head_fea), dim=0), gate_list

    def __repr__(self):
        return self.__class__.__name__


class MessageLayer(nn.Module):
    """
    Massage Layers are used to propagate information between nodes in
    the stoichiometry graph.
    """

    def __init__(self, elem_fea_len, edge_fea_len, elem_heads, elem_gate, elem_msg):
        """
        """
        super().__init__()

        # Pooling and Output
        self.edge_ebd = SimpleNetwork(edge_fea_len, 128, [edge_fea_len,128,128,128])
        self.pooling = nn.ModuleList(
            [
                WeightedAttentionPooling(
                    gate_nn=SimpleNetwork(2 * elem_fea_len + 128, 1, elem_gate),
                    message_nn=SimpleNetwork(2 * elem_fea_len + 128, elem_fea_len, elem_msg),
                )
                for _ in range(elem_heads)
            ]
        )
        

    def forward(self, elem_weights, elem_in_fea, edge_fea, self_fea_idx, nbr_fea_idx):
        """
        Forward pass

        Parameters
        ----------
        N: Total number of elements (nodes) in the batch
        M: Total number of pairs (edges) in the batch
        C: Total number of crystals (graphs) in the batch

        Inputs
        ----------
        elem_weights: Variable(torch.Tensor) shape (N,)
            The fractional weights of elems in their materials
        elem_in_fea: Variable(torch.Tensor) shape (N, elem_fea_len)
            Element hidden features before message passing
        self_fea_idx: torch.Tensor shape (M,)
            Indices of the first element in each of the M pairs
        nbr_fea_idx: torch.Tensor shape (M,)
            Indices of the second element in each of the M pairs

        Returns
        -------
        elem_out_fea: nn.Variable shape (N, elem_fea_len)
            Element hidden features after message passing
        """
        # construct the total features for passing
        elem_nbr_weights = elem_weights[nbr_fea_idx, :]
        elem_nbr_fea = elem_in_fea[nbr_fea_idx, :]
        elem_self_fea = elem_in_fea[self_fea_idx, :]
        edge_fea = self.edge_ebd(edge_fea)
        fea = torch.cat([elem_self_fea, elem_nbr_fea, edge_fea], dim=1)

        # sum selectivity over the neighbours to get elems
        head_fea = []
        gate_list = []
        for attnhead in self.pooling:
            out, gate = attnhead(fea, index=self_fea_idx, weights=elem_nbr_weights)
            head_fea.append(out)
            gate_list.append(gate)

        # average the attention heads
        fea = torch.mean(torch.stack(head_fea), dim=0)

        return fea + elem_in_fea, gate_list

    def __repr__(self):
        return self.__class__.__name__
