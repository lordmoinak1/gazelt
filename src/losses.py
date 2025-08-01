import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

def get_loss(args, weights, train_dataset):
    if args.loss == 'ce':
        loss_fxn = torch.nn.CrossEntropyLoss(weight=weights, reduction='mean')
    elif args.loss == 'focal':
        loss_fxn = torch.hub.load('adeelh/pytorch-multi-class-focal-loss', model='FocalLoss', alpha=weights, gamma=args.fl_gamma, reduction='mean')
    elif args.loss == 'ldam':
        loss_fxn = LDAMLoss(cls_num_list=train_dataset.cls_num_list, weight=weights)

    return loss_fxn

def get_CB_weights(samples_per_cls, beta):
    effective_num = 1.0 - np.power(beta, samples_per_cls)

    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * len(samples_per_cls)

    return weights

## CREDIT TO https://github.com/kaidic/LDAM-DRW ##
class LDAMLoss(torch.nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

        print(self.weight)

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)

        return F.cross_entropy(self.s*output, target, weight=self.weight)

def global_loss(x, y):
    return (F.normalize(x.pow(2).mean(1).view(x.size(0), -1)) - F.normalize(y.pow(2).mean(1).view(y.size(0), -1))).pow(2).mean()

def focal_loss(x, y):
    return (F.normalize(x.pow(2).mean(1).view(x.size(0), -1)) - F.normalize(y.pow(2).mean(1).view(y.size(0), -1))).pow(2).mean()

def bhattacharyya_loss(student, focal_teacher, global_teacher):
    lambda_0 = 0.5
    lambda_1 = 0.5
    out_focal = nn.KLDivLoss(reduction="batchmean", log_target=True)(F.log_softmax(student, dim=1), F.log_softmax(focal_teacher, dim=1))
    out_global = nn.KLDivLoss(reduction="batchmean", log_target=True)(F.log_softmax(student, dim=1), F.log_softmax(global_teacher, dim=1))
    return lambda_0*out_focal+lambda_1*out_global
