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
from model import SSNGNN
import json

def parse_list(input_str):
    return list(map(int, input_str.strip('[]').split(',')))

parser = argparse.ArgumentParser(description='Solid Solution Nested Graph Neural Networks')
# parser.add_argument('data_options', metavar='OPTIONS', nargs='+',
#                     help='dataset options, started with the path to root dir, '
#                          'then other options')

parser.add_argument('--train_data')
parser.add_argument('--val_data')
parser.add_argument('--embedding')
parser.add_argument('--savepath')
parser.add_argument('--seed', default=581)
parser.add_argument('--gamma', default=0.1)
parser.add_argument('--task', choices=['regression', 'classification'],
                    default='regression', help='complete a regression or '
                                                   'classification task (default: regression)')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run (default: 100)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
# parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
#                     metavar='LR', help='initial learning rate (default: '
#                                        '0.01)')
parser.add_argument('--lr-milestones', default=[30, 60, 90, 120, 150, 180, 210, 240, 270 ,300], type=parse_list)
# parser.add_argument('--lr-milestones', default=[10,20,30,40,50,60,70,80,90,100], type=parse_list)
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay (default: 0)')
# parser.add_argument('--print-freq', '-p', default=10, type=int,
#                     metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--optim', default='AdamW', type=str, metavar='AdamW',
                    help='choose an optimizer, SGD/Adam/AdamW, (default: AdamW)')

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

if args.task == 'regression':
    best_mae_error = 1e10
else:
    best_mae_error = 0.


def main():
    global args, best_mae_error
    # seed = 581
    torch.manual_seed(args.seed)
    # load data
    train_dataset = SSDataset(args.train_data, args.embedding)
    val_dataset = SSDataset(args.val_data, args.embedding)
    collate_fn = collate_batch
    
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size,
                              collate_fn = collate_batch, shuffle = True)
    
    val_loader = DataLoader(val_dataset, batch_size = args.batch_size,
                              collate_fn = collate_batch, shuffle = False)
    
    if args.task == 'classification':
        normalizer = Normalizer(torch.zeros(2))
        normalizer.load_state_dict({'mean': 0., 'std': 1.})

    sample_data_list = [train_dataset[i] for i in range(len(train_dataset))]
       
    sample_target = collate_fn(sample_data_list)[9]
    normalizer = Normalizer(sample_target)

    # build model
    comp_fea_len = train_dataset[0][1].shape[-1]
    edge_fea_len = train_dataset[0][2].shape[-1]
    bond_fea_len = train_dataset[0][5].shape[-1]
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

    # define loss func and optimizer
    if args.task == 'classification':
        criterion = nn.NLLLoss()
    else:
        criterion = nn.MSELoss()
        
   
    optimizer = optim.AdamW([{'params':model.comp_gnn.parameters(), 'lr': 1e-6, 'weight_decay': 1},
                             {'params':model.fc.parameters(), 'lr':1e-6, 'weight_decay': 1},
                             {'params':model.struct_gnn.parameters(), 'lr': 0.001, 'weight_decay': 1}])

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_mae_error = checkpoint['best_mae_error']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            normalizer.load_state_dict(checkpoint['normalizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    scheduler = MultiStepLR(optimizer, milestones=args.lr_milestones,
                            gamma=args.gamma)
    
    train_loss_dict, train_mae_dict, val_loss_dict, val_mae_dict = {}, {}, {}, {}
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        loss, mae = train(train_loader, model, criterion, optimizer, epoch, normalizer)
        train_loss_dict[epoch] = loss
        train_mae_dict[epoch] = mae
        # evaluate on validation set
        loss, mae = validate(val_loader, model, criterion, normalizer)
        val_loss_dict[epoch] = loss
        val_mae_dict[epoch] = mae
        with open(args.savepath +'train_loss.json', 'w', encoding='utf-8') as json_file:
            json.dump(train_loss_dict, json_file, ensure_ascii=True, indent=4)
            json_file.close()
        with open(args.savepath + 'train_mae.json', 'w', encoding='utf-8') as json_file:
            json.dump(train_mae_dict, json_file, ensure_ascii=True, indent=4)
            json_file.close()
        with open(args.savepath + 'val_loss.json', 'w', encoding='utf-8') as json_file:
            json.dump(val_loss_dict, json_file, ensure_ascii=True, indent=4)
            json_file.close()
        with open(args.savepath + 'val_mae.json', 'w', encoding='utf-8') as json_file:
            json.dump(val_mae_dict, json_file, ensure_ascii=True, indent=4)
            json_file.close()
        

        scheduler.step()

       
        filename = args.savepath +str(epoch) + 'checkpoint.pth.tar'
        torch.save({'epoch':epoch,
                   'state_dict': model.state_dict(),
                   'optimizer': optimizer.state_dict(),
                   'normalizer': normalizer.state_dict(),
                   'args': vars(args)}
                   , filename)

    best_mae = 1e10
    for e in val_mae_dict.keys():
        if val_mae_dict[e] < best_mae:
            best_mae = val_mae_dict[e]
            epoch = e
    print(epoch, best_mae)
    
    best_name = args.savepath +str(epoch) + 'checkpoint.pth.tar'
    
    shutil.copy(best_name, args.savepath + 'best.pth.tar')
           
   
        
    
    
        

def train(train_loader, model, criterion, optimizer, epoch, normalizer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    if args.task == 'regression':
        mae_errors = AverageMeter()
    else:
        accuracies = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()
        fscores = AverageMeter()
        auc_scores = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, batch_data in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.cuda:
            device = torch.device('cuda:0')
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
        else:
            comp_weights, comp_fea, edge_fea, self_fea_idx, comp_nbr_fea_idx, comp_node_idx,\
                struct_nbr_fea, struct_nbr_fea_idx, struct_node_idx, target = batch_data
            input_var = (comp_weights, comp_fea, edge_fea, self_fea_idx, comp_nbr_fea_idx, comp_node_idx,\
                struct_nbr_fea, struct_nbr_fea_idx, struct_node_idx)
        # normalize target
        if args.task == 'regression':
            target_normed = normalizer.norm(target)
        else:
            target_normed = target.view(-1).long()
        if args.cuda:
            target_var = Variable(target_normed.cuda(non_blocking=True))
        else:
            target_var = Variable(target_normed)

        # compute output
        output,_ = model(*input_var)
        loss = criterion(output, target_var)
            
        # measure accuracy and record loss
        if args.task == 'regression':
            mae_error = mae(normalizer.denorm(output.data), target)
            losses.update(loss.data.cpu().item(), target.size(0))
            mae_errors.update(mae_error.cpu().item(), target.size(0))
        else:
            accuracy, precision, recall, fscore, auc_score = \
                class_eval(output.data.cpu(), target)
            losses.update(loss.data.cpu().item(), target.size(0))
            accuracies.update(accuracy, target.size(0))
            precisions.update(precision, target.size(0))
            recalls.update(recall, target.size(0))
            fscores.update(fscore, target.size(0))
            auc_scores.update(auc_score, target.size(0))
        

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # if i % args.print_freq == 0:
        if True:
        # if loss.data.cpu().item()>1:
            if args.task == 'regression':
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, mae_errors=mae_errors)
                )
            else:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accu {accu.val:.3f} ({accu.avg:.3f})\t'
                      'Precision {prec.val:.3f} ({prec.avg:.3f})\t'
                      'Recall {recall.val:.3f} ({recall.avg:.3f})\t'
                      'F1 {f1.val:.3f} ({f1.avg:.3f})\t'
                      'AUC {auc.val:.3f} ({auc.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, accu=accuracies,
                    prec=precisions, recall=recalls, f1=fscores,
                    auc=auc_scores)
                )
    if args.task == 'regression':
        return losses.avg, mae_errors.avg
    else:
        return losses.avg, auc_scores.avg

def validate(val_loader, model, criterion, normalizer, test=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    if args.task == 'regression':
        mae_errors = AverageMeter()
    else:
        accuracies = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()
        fscores = AverageMeter()
        auc_scores = AverageMeter()
    if test:
        test_targets = []
        test_preds = []
        test_cif_ids = []

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, batch_data in enumerate(val_loader):
        if args.cuda:
            with torch.no_grad():
                device = torch.device('cuda:0')
                comp_weights, comp_fea,edge_fea, self_fea_idx, comp_nbr_fea_idx, comp_node_idx,\
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
        else:
            with torch.no_grad():
                comp_weights, comp_fea, edge_fea, self_fea_idx, comp_nbr_fea_idx, comp_node_idx,\
                    struct_nbr_fea, struct_nbr_fea_idx, struct_node_idx, target = batch_data
                input_var = (comp_weights, comp_fea, edge_fea, self_fea_idx, comp_nbr_fea_idx, comp_node_idx,\
                    struct_nbr_fea, struct_nbr_fea_idx, struct_node_idx)
        if args.task == 'regression':
            target_normed = normalizer.norm(target)
        else:
            target_normed = target.view(-1).long()
        if args.cuda:
            with torch.no_grad():
                target_var = Variable(target_normed.cuda(non_blocking=True))
        else:
            with torch.no_grad():
                target_var = Variable(target_normed)

        # compute output
        output,_ = model(*input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        if args.task == 'regression':
            mae_error = mae(normalizer.denorm(output.data), target)
            losses.update(loss.data.cpu().item(), target.size(0))
            mae_errors.update(mae_error.cpu().item(), target.size(0))
            if test:
                test_pred = normalizer.denorm(output.data.cpu())
                test_target = target
                test_preds += test_pred.view(-1).tolist()
                test_targets += test_target.view(-1).tolist()
                # test_cif_ids += batch_cif_ids
        else:
            accuracy, precision, recall, fscore, auc_score = \
                class_eval(output.data.cpu(), target)
            losses.update(loss.data.cpu().item(), target.size(0))
            accuracies.update(accuracy, target.size(0))
            precisions.update(precision, target.size(0))
            recalls.update(recall, target.size(0))
            fscores.update(fscore, target.size(0))
            auc_scores.update(auc_score, target.size(0))
            if test:
                test_pred = torch.exp(output.data.cpu())
                test_target = target
                assert test_pred.shape[1] == 2
                test_preds += test_pred[:, 1].tolist()
                test_targets += test_target.view(-1).tolist()
                # test_cif_ids += batch_cif_ids

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # if i % args.print_freq == 0:
        if True:
            if args.task == 'regression':
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    mae_errors=mae_errors))
            else:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accu {accu.val:.3f} ({accu.avg:.3f})\t'
                      'Precision {prec.val:.3f} ({prec.avg:.3f})\t'
                      'Recall {recall.val:.3f} ({recall.avg:.3f})\t'
                      'F1 {f1.val:.3f} ({f1.avg:.3f})\t'
                      'AUC {auc.val:.3f} ({auc.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    accu=accuracies, prec=precisions, recall=recalls,
                    f1=fscores, auc=auc_scores))

    if test:
        star_label = '**'
        import csv
        with open('test_results.csv', 'w') as f:
            writer = csv.writer(f)
            for cif_id, target, pred in zip(test_cif_ids, test_targets,
                                            test_preds):
                writer.writerow((cif_id, target, pred))
    else:
        star_label = '*'
    if args.task == 'regression':
        print(' {star} MAE {mae_errors.avg:.3f}'.format(star=star_label,
                                                        mae_errors=mae_errors))
        return losses.avg, mae_errors.avg
    else:
        print(' {star} AUC {auc.avg:.3f}'.format(star=star_label,
                                                 auc=auc_scores))
        return losses.avg, auc_scores.avg

class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


def mae(prediction, target):
    """
    Computes the mean absolute error between prediction and target

    Parameters
    ----------

    prediction: torch.Tensor (N, 1)
    target: torch.Tensor (N, 1)
    """
    return torch.mean(torch.abs(target - prediction))


def class_eval(prediction, target):
    prediction = np.exp(prediction.numpy())
    target = target.numpy()
    pred_label = np.argmax(prediction, axis=1)
    target_label = np.squeeze(target)
    if not target_label.shape:
        target_label = np.asarray([target_label])
    if prediction.shape[1] == 2:
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            target_label, pred_label, average='binary')
        auc_score = metrics.roc_auc_score(target_label, prediction[:, 1])
        accuracy = metrics.accuracy_score(target_label, pred_label)
    else:
        raise NotImplementedError
    return accuracy, precision, recall, fscore, auc_score


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def adjust_learning_rate(optimizer, epoch, k):
    """Sets the learning rate to the initial LR decayed by 10 every k epochs"""
    assert type(k) is int
    lr = args.lr * (0.1 ** (epoch // k))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
