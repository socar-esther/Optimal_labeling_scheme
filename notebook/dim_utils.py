import os 
import numpy as np

import argparse
import socket
import time
import sys
import pandas as pd 
from sklearn.cluster import KMeans

import torchvision
from torchvision import transforms
from torchvision.models import resnet50
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import copy

import config
from utils import *
import csv
from argparse import ArgumentParser
import json

def get_dim_parser(opt) : 
    # get config
    parser = argparse.ArgumentParser(description='Dimension estimation')
    parser.add_argument('--save_dir', default='dim_outputs/cifar10/', help="dataset to use for dim estimation")
    parser.add_argument('--model', default='resnet50', help="model to do dimension estimation on")
    parser.add_argument('--pretrained', default=True, help="whether pre-trained flag is true")
    parser.add_argument('--n_factors', default=3, help="number of factors (including residual)")
    parser.add_argument('--residual_index', default=2, help="index of residual factor (usually last)")
    parser.add_argument('--batch_size', default=4, help="batch size during evaluation")
    parser.add_argument('--image_size', default=256, type=int, help="image size during evaluation")
    parser.add_argument('--num_workers', default=4, help="number of CPU threads")
    parser.add_argument('--device', default='cuda:0', help="gpu id")
    parser.add_argument('--dataset', default='stylized_cifar10')
    parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                            help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
    parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                            help='convert model torchscript for inference')
    args = parser.parse_args()


    if not os.path.exists(args.save_dir) : 
        os.mkdir(args.save_dir)
        
    # get trained classifier
    print('-- Loading trained classifier for final score')
    model = get_model(args)

    device = args.device
    model.cuda(device)
    model = model.eval()

    args.n_class = 10
    args.model_path = './checkpoint/cifar10_best.pt'
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, args.n_class)

    model.load_state_dict(torch.load(opt.model_path)) 
    
    return args, model

def dim_est(output_dict, factor_list, args) :
    
    # step1. get vectors from the input pair images
    za = np.concatenate(output_dict['example1'])
    zb = np.concatenate(output_dict['example2'])
    factors = np.concatenate(factor_list)
    
    # step2. factor(shape, texture)별로 다시 계산 for score
    za_by_factor = dict()
    zb_by_factor = dict()
    mean_by_factor = dict()
    score_by_factor = dict()
    individual_scores = dict()
    
    zall = np.concatenate([za, zb], 0)
    mean = np.mean(zall, 0, keepdims = True)
    var = np.sum(np.mean((zall-mean)*(zall-mean), 0))
    
    ## step 2-1. 각 factor별로 돌면서 다시 계산 
    for f in range(args.n_factors) : 
        if f != args.residual_index : 
            indices = np.where(factors == f)[0]
            
            za_by_factor[f] = za[indices]
            zb_by_factor[f] = zb[indices]
            mean_by_factor[f] = 0.5*(np.mean(za_by_factor[f], 0, keepdims=True)+np.mean(zb_by_factor[f], 0, keepdims=True))
            
            score_by_factor[f] = np.sum(np.mean((za_by_factor[f]-mean_by_factor[f])*(zb_by_factor[f]-mean_by_factor[f]), 0))
            score_by_factor[f] = score_by_factor[f]/var
            
            idv = np.mean((za_by_factor[f]-mean_by_factor[f])*(zb_by_factor[f]-mean_by_factor[f]), 0)/var
            individual_scores[f] = idv
        
        else : 
            # residual factor에 해당되는 경우
            score_by_factor[f] = 1.0
    
    scores = np.array([score_by_factor[f] for f in range(args.n_factors) ])
    print(score_by_factor) 
    
    # step3. softmax output (shape(0) or texture(1)); for the shape
    m = np.max(scores)
    
    e = np.exp(scores-m)
    softmaxed = e/np.sum(e)
    
    dim = za.shape[1]
    dims = [int(s*dim) for s in softmaxed]
    
    dims[-1] = dim - sum(dims[:-1])
    dims_percent = dims.copy()
    
    for i in range(len(dims)) : 
        dims_percent[i] = round(100*(dims[i] / sum(dims,)), 1)
    return dims, dims_percent