# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 15:44:05 2024

@author: YidingWang
"""

# import torch
import torch.nn as nn
import torch.nn.functional as F

from comp_gnn import DescriptorNetwork
from roost_segments import SimpleNetwork, ResidualNetwork
from struct_gnn import CrystalGraphConvNet



class SSNGNN(nn.Module):
    def __init__(self,
                 comp_fea_len,
                 elem_fea_len = 128,#comp_fea经过embedding的长度
                 edge_fea_len = 2,
                 n_comp_mp_layers = 3,#comp graph消息传递层数
                 mp_heads = 3, #消息传递层每层几个head
                 mp_gate = [256],#消息传递层中f的中间层结构
                 mp_msg = [256], #消息传递层中g的中间层结构
                 pooling_heads = 3, #池化层每层几个head
                 pooling_gate = [256],#池化层中f的中间层结构
                 pooling_msg = [256], #池化层中g的中间层结构
                 atom_fea_len = 128, #struct graph节点的特征长度
                 fc_hidden_dims = [128],#fully connected中间层结构
                 bond_fea_len = 41, #stuct graph中的边特征长度
                 atom_hidden_fea_len = 64,#Number of hidden atom features in the convolutional layers
                 n_struct_conv_layers = 3,#Number of convolutional layers
                 h_fea_len = 128,#Number of hidden features after pooling
                 n_h = 1,#Number of hidden layers after pooling
                 classification=False,
                 get_embedding = False
                 ):
        super(SSNGNN,self).__init__()
        self.comp_gnn = DescriptorNetwork(elem_emb_len=comp_fea_len,
                                            elem_fea_len=elem_fea_len,
                                            edge_fea_len=edge_fea_len,
                                            n_graph=n_comp_mp_layers,
                                            elem_heads=mp_heads,
                                            elem_gate=mp_gate,
                                            elem_msg=mp_msg,
                                            cry_heads=pooling_heads,
                                            cry_gate=pooling_gate,
                                            cry_msg=pooling_msg)
        self.fc = SimpleNetwork(elem_fea_len,
                                atom_fea_len,
                                fc_hidden_dims,
                                # activation= nn.Sigmoid,
                                batchnorm=True)
        self.bn = nn.BatchNorm1d(atom_fea_len)
        self.dropout = nn.Dropout(p=0.7)

        self.struct_gnn = CrystalGraphConvNet(orig_atom_fea_len=atom_fea_len,
                                              nbr_fea_len=bond_fea_len,
                                              atom_fea_len=atom_hidden_fea_len,
                                              n_conv=n_struct_conv_layers,
                                              h_fea_len=h_fea_len,
                                              n_h=n_h,
                                              classification=classification,
                                              get_embedding = get_embedding)
    
    def forward(self,
                 comp_weights,
                 comp_fea,
                 edge_fea,
                 self_fea_idx,
                 comp_nbr_fea_idx,
                 comp_node_idx,
                 struct_nbr_fea,
                 struct_nbr_fea_idx,
                 struct_node_idx):
        struct_node_fea, gate_list = self.comp_gnn(comp_weights,
                                        comp_fea,
                                        edge_fea,
                                        self_fea_idx,
                                        comp_nbr_fea_idx,
                                        comp_node_idx)

        struct_node_fea = self.fc(struct_node_fea)

        out = self.struct_gnn(struct_node_fea,
                              struct_nbr_fea,
                              struct_nbr_fea_idx,
                              struct_node_idx)
        return out, gate_list
    
    
    