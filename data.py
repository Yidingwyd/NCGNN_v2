# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 17:46:22 2024

@author: YidingWang
"""

from pymatgen.core import Composition, Lattice, Structure
import json, warnings, torch, tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader


class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.

    Unit: angstrom
    """
    def __init__(self, dmin, dmax, step, var=None):
        """
        Parameters
        ----------

        dmin: float
          Minimum interatomic distance
        dmax: float
          Maximum interatomic distance
        step: float
          Step size for the Gaussian filter
        """
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        """
        Apply Gaussian disntance filter to a numpy distance array

        Parameters
        ----------

        distance: np.array shape n-d array
          A distance matrix of any shape

        Returns
        -------
        expanded_distance: shape (n+1)-d array
          Expanded distance matrix with the last dimension of length
          len(self.filter)
        """
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 /
                      self.var**2)

class SSDataset(Dataset):
    def __init__(self, data_path, embedding_file, max_num_nbr = 12, radius = 10,
                 dmin=0, step=1):
        self.max_num_nbr = max_num_nbr
        self.radius = radius
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)
        with open(data_path, 'r', encoding='utf-8') as json_file:
            self.cpa_dataset = json.load(json_file)
            json_file.close()
        with open(embedding_file) as f:
            self.embedding = json.load(f)
            f.close()
        self.final_dataset = []
        for key in tqdm.tqdm(self.cpa_dataset.keys(), ncols= 50):
            cpa_data = self.cpa_dataset[key]
            self.final_dataset.append(self.get_feas(cpa_data, key))
            # if eval(key) >= 10:
            #     break
    def get_feas(self, cpa_data, key):
        comp_dict = cpa_data['comp']
        comp_weights, comp_fea, self_fea_idx, comp_nbr_fea_idx, edge_fea, formula_list  = [], [], [], [], [],[]
        M_list = []
        for k in comp_dict.keys():
            comp_graph = comp_dict[k]
            # composition = Composition(formula)
            # elements = composition.elements
            elements = list(comp_graph.keys())
            M_list.append(len(elements))
            # weights = [[composition.get_atomic_fraction(e)] for e in elements]
            weights = [[comp_graph[e]['fraction']] for e in elements]
            for e in elements:
                edge = comp_graph[e]['edge']
                edge_fea = edge_fea + edge
            comp_weights = comp_weights + weights
            
            for e in elements:
                # comp_fea.append(self.embedding[e.symbol])
                comp_fea.append(self.embedding[e])
            
            assert len(elements) == len(weights)
            formula = ''
            for e, w in zip(elements, weights):
                formula = formula + e + str(w[0])
            formula_list.append(formula)
            
            if self_fea_idx == []:
                n = 0
            else:
                n = self_fea_idx[-1]+1
            self_fea, nbr_fea = [], []
            # count = 0
            for m1 in range(len(elements)):
                for m2 in range(len(elements)):
                    self_fea.append(m1)
                    nbr_fea.append(m2)
                    # edge_fea.append(count)
                    # count += 1
            self_fea_idx = self_fea_idx + [x+n for x in self_fea]
            comp_nbr_fea_idx = comp_nbr_fea_idx + [x+n for x in nbr_fea]
            # edge_fea_idx = edge_fea_idx + [x+m for x in edge_fea]

        lattice = Lattice.from_parameters(cpa_data['a'],
                                          cpa_data['b'],
                                          cpa_data['c'],
                                          cpa_data['alpha'],
                                          cpa_data['beta'],
                                          cpa_data['gamma'])
        try:
            crystal = Structure(lattice,
                                  species = list(range(1, len(comp_dict)+1)),
                                  coords = [eval(coord) for coord in comp_dict.keys()])
        except:
            print(lattice)
            print(comp_dict)

        
        all_nbrs = crystal.get_all_neighbors(r=self.radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        struct_nbr_fea_idx, struct_nbr_fea = [], []
        for nbr in all_nbrs:
            if len(nbr) < self.max_num_nbr:
                warnings.warn('{} not find enough neighbors to build graph. '
                              'If it happens frequently, consider increase '
                              'radius.'.format(key))
                struct_nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +
                                   [0] * (self.max_num_nbr - len(nbr)))
                struct_nbr_fea.append(list(map(lambda x: x[1], nbr)) +
                               [self.radius + 1.] * (self.max_num_nbr -
                                                     len(nbr)))
            else:
                struct_nbr_fea_idx.append(list(map(lambda x: x[2],
                                            nbr[:self.max_num_nbr])))
                struct_nbr_fea.append(list(map(lambda x: x[1],
                                        nbr[:self.max_num_nbr])))
        struct_nbr_fea = np.array(struct_nbr_fea)
        struct_nbr_fea = self.gdf.expand(struct_nbr_fea)
        return (
                    torch.Tensor(comp_weights),
                    torch.Tensor(comp_fea),
                    torch.Tensor(edge_fea),
                    torch.LongTensor(self_fea_idx),
                    torch.LongTensor(comp_nbr_fea_idx),
                    torch.Tensor(struct_nbr_fea),
                    torch.LongTensor(struct_nbr_fea_idx),
                    torch.Tensor([cpa_data['target']]),
                    M_list
                )
    def __len__(self):
        return len(self.final_dataset)
    def __getitem__(self, i):
        return self.final_dataset[i]


def collate_batch(dataset_list):
    batch_comp_weights = []
    batch_comp_fea = []
    batch_edge_fea = []
    batch_self_fea_idx = []
    batch_comp_nbr_fea_idx = []
    comp_node_idx = []
    batch_struct_nbr_fea = []
    batch_struct_nbr_fea_idx = []
    struct_node_idx = []
    batch_target = []
    
    comp_base_idx = 0
    struct_base_idx = 0
    comp_graph_count = 0
    for i, (comp_weights, comp_fea, edge_fea, self_fea_idx, comp_nbr_fea_idx, \
        struct_nbr_fea, struct_nbr_fea_idx, target, M_list) in enumerate(dataset_list):
        
        batch_comp_weights.append(comp_weights)
        batch_comp_fea.append(comp_fea)
        batch_edge_fea.append(edge_fea)
        batch_self_fea_idx.append(self_fea_idx + comp_base_idx)
        batch_comp_nbr_fea_idx.append(comp_nbr_fea_idx + comp_base_idx)
        for M in M_list:
            comp_node_idx.append(torch.LongTensor([comp_graph_count]*M))
            comp_base_idx += M
            comp_graph_count += 1
        
        batch_struct_nbr_fea.append(struct_nbr_fea)
        batch_struct_nbr_fea_idx.append(struct_nbr_fea_idx + struct_base_idx)
        N = struct_nbr_fea.shape[0]
        new_idx = torch.LongTensor(np.arange(N)+struct_base_idx)
        struct_node_idx.append(new_idx)
        struct_base_idx += N
        
        batch_target.append(target)
     
    return (
                torch.cat(batch_comp_weights, dim = 0),
                torch.cat(batch_comp_fea, dim = 0),
                torch.cat(batch_edge_fea, dim = 0),
                torch.cat(batch_self_fea_idx, dim = 0),
                torch.cat(batch_comp_nbr_fea_idx, dim = 0),
                torch.cat(comp_node_idx, dim = 0),
                torch.cat(batch_struct_nbr_fea, dim = 0),
                torch.cat(batch_struct_nbr_fea_idx, dim = 0),
                struct_node_idx,
                torch.stack(batch_target, dim=0)
            )
     



