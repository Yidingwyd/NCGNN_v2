import argparse
import os
import shutil
import sys
import time
import warnings
from random import sample

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, random_split
from data import SSDataset, collate_batch
# from cgcnn_data import  get_train_val_test_loader
from model import SSNGNN
import json
import pandas as pd

def parse_list(input_str):
    return list(map(float, input_str.strip('[]').split(',')))

parser = argparse.ArgumentParser(description='Solid Solution Nested Graph Neural Networks')
# parser.add_argument('data_options', metavar='OPTIONS', nargs='+',
#                     help='dataset options, started with the path to root dir, '
#                          'then other options')


parser.add_argument('--test_data')
parser.add_argument('--embedding')
parser.add_argument('--savepath')
parser.add_argument('--modelpath')
parser.add_argument('--seed', default=581)
parser.add_argument('--gamma', default=0.7)
parser.add_argument('--task', choices=['regression', 'classification'],
                    default='regression', help='complete a regression or '
                                                   'classification task (default: regression)')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--elem-fea-len', default=128, type=int, metavar='N')
parser.add_argument('--n-comp-mp-layers', default=3, type=int, metavar='N')
parser.add_argument('--mp-heads', default=3, type=int, metavar='N')
parser.add_argument('--mp-gate', default=[256], type=parse_list)
parser.add_argument('--mp-msg', default=[256], type=parse_list)
parser.add_argument('--pooling-heads', default=3, type=int, metavar='N')
parser.add_argument('--pooling-gate', default=[256], type=parse_list)
parser.add_argument('--pooling-msg', default=[256], type=parse_list)
parser.add_argument('--atom-fea-len', default=128, type=int, metavar='N',
                    help='number of hidden atom features in conv layers')
parser.add_argument('--fc-hidden-dims', default=[128], type=parse_list)
parser.add_argument('--atom-hidden-fea-len', default=64, type=int, metavar='N')
parser.add_argument('--n-struct-conv-layers', default=3, type=int, metavar='N')
parser.add_argument('--h-fea-len', default=128, type=int, metavar='N',
                    help='number of hidden features after pooling')
parser.add_argument('--n-h', default=1, type=int, metavar='N',
                    help='number of hidden layers after pooling')


args = parser.parse_args(sys.argv[1:])
args.cuda = not args.disable_cuda and torch.cuda.is_available()

def main():
    global args, best_mae_error
    dataset = SSDataset(args.test_data, args.embedding)
    
    test_loader = DataLoader(dataset, batch_size = args.batch_size,
                              collate_fn = collate_batch, shuffle = False)
    
    comp_fea_len = dataset[0][1].shape[-1]
    edge_fea_len = dataset[0][2].shape[-1]
    bond_fea_len = dataset[0][5].shape[-1]
    model = SSNGNN(comp_fea_len = comp_fea_len,
                    elem_fea_len = args.elem_fea_len,#comp_fea经过embedding的长度
                    edge_fea_len = edge_fea_len,
                    n_comp_mp_layers = args.n_comp_mp_layers,#comp graph消息传递层数
                    mp_heads = args.mp_heads, #消息传递层每层几个head
                    mp_gate = args.mp_gate,#消息传递层中f的中间层结构
                    mp_msg = args.mp_msg, #消息传递层中g的中间层结构
                    pooling_heads = args.pooling_heads, #池化层每层几个head
                    pooling_gate = args.pooling_gate,#池化层中f的中间层结构
                    pooling_msg = args.pooling_msg, #池化层中g的中间层结构
                    atom_fea_len = args.atom_fea_len, #struct graph节点的特征长度
                    fc_hidden_dims = args.fc_hidden_dims,#fully connected中间层结构
                    bond_fea_len = bond_fea_len, #stuct graph中的边特征长度
                    atom_hidden_fea_len = args.atom_hidden_fea_len,#Number of hidden atom features in the convolutional layers
                    n_struct_conv_layers = args.n_struct_conv_layers,#Number of convolutional layers
                    h_fea_len = args.h_fea_len,#Number of hidden features after pooling
                    n_h = args.n_h,#Number of hidden layers after pooling
                    classification=True if args.task ==
                                           'classification' else False)
    if args.cuda:
        model.cuda()  
    normalizer = Normalizer(torch.zeros(3))
    device = torch.device('cuda:0') if args.cuda else torch.device('cpu')
    # print(device)
    if os.path.isfile(args.modelpath):
        print("=> loading model '{}'".format(args.modelpath))
        checkpoint = torch.load(args.modelpath,
                                map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        normalizer.load_state_dict(checkpoint['normalizer'], device)

    else:
        print("=> no model found at '{}'".format(args.modelpath))
        
    formula_list, output_list, target_list = [],[],[]
    model.eval()
    for i, batch_data in enumerate(test_loader):
        # print(i)
        # if args.cuda:
        with torch.no_grad():
            device = torch.device('cuda:0') if args.cuda else torch.device('cpu')
            comp_weights, comp_fea, edge_fea, self_fea_idx, comp_nbr_fea_idx, comp_node_idx,\
                struct_nbr_fea, struct_nbr_fea_idx, struct_node_idx, target = batch_data
            comp_weights = comp_weights.to(device)
            comp_fea = comp_fea.to(device)
            edge_fea = edge_fea.to(device)
            self_fea_idx = self_fea_idx.to(device)
            comp_nbr_fea_idx = comp_nbr_fea_idx.to(device)
            comp_node_idx = comp_node_idx.to(device)
            struct_nbr_fea = struct_nbr_fea.to(device)
            struct_nbr_fea_idx = struct_nbr_fea_idx.to(device)
            struct_node_idx = [idx.to(device) for idx in struct_node_idx]
            target = target.to(device)
            input_var = (comp_weights, comp_fea, edge_fea, self_fea_idx, comp_nbr_fea_idx, comp_node_idx,\
                struct_nbr_fea, struct_nbr_fea_idx, struct_node_idx)
        
            output,_ = model(*input_var)
            # print(output.data)
            output = normalizer.denorm(output.data)
            # print(output)
            output = output.squeeze().tolist() 
            output = output if type(output) is list else [output]
            target = target.squeeze().tolist()
            target = target if type(target) is list else [target]
            # formula_list = formula_list + composition_list
            output_list = output_list + output
            target_list = target_list + target
    
    with open(args.test_data, 'r', encoding='utf-8') as json_file:
        data_dict = json.load(json_file)
        json_file.close()
    assert len(output_list) == len(data_dict)
    
    # for d in list(data_dict.values()):
    #     formula = list(d['comp'].values())[0]
    #     # target = d['target']
    #     formula_list.append(formula)
    
    df = pd.DataFrame({
                       'output':output_list,
                       'target':target_list})
    df.to_excel(args.savepath)




class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor,0,True).to(tensor.device)
        self.std = torch.std(tensor,0,True).to(tensor.device)

    def norm(self, tensor):
        # print(tensor.device)
        # print(self.mean.device)
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        # print(normed_tensor.device)
        # print(self.std.device)
        # print(self.mean.device)
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict, device):
        self.mean = state_dict['mean'].to(device)
        self.std = state_dict['std'].to(device)

if __name__ == '__main__':

    main()