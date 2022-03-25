from __future__ import print_function

import os
import argparse
import socket
import time
import sys

import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import torchvision
from util import adjust_learning_rate, accuracy, AverageMeter, accuracy
from torchvision import transforms
from torchvision.models import resnet50
from sklearn.metrics import f1_score

from dataloader import get_datasets
from train import train, validate
from test import test

print('-- check torch version : ', torch.__version__)    # 1.7.x


def parse_option() :

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--eval_freq', type=int, default=10, help='meta-eval frequency')
    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=10, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=4, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,80,100,120,140,160,180', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--adam', action='store_true', help='use adam optimizer')

    # dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--n_class', type=int, default=10)
    parser.add_argument('--n_style', type=list, default=['style1', 'style2', 'style3', 'style4'])
    parser.add_argument('--train_root', type=str, default='../../shared/users/esther/')
    parser.add_argument('--test_root', type=str, default='../../shared/users/esther/')

    # cosine annealing
    parser.add_argument('--cosine', action='store_true', help='using cosine annealing')

    # specify folder
    parser.add_argument('--model_path', type=str, default='', help='path to save model')
    parser.add_argument('--tb_path', type=str, default='', help='path to tensorboard')
    parser.add_argument('--data_root', type=str, default='', help='path to data root')

    opt = parser.parse_args()
    
    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.n_gpu = torch.cuda.device_count()
    
    return opt


def main() :
    
    opt = parse_option()
    
    for i in range(len(opt.n_style)) : 
        
        print('-- style : ', opt.n_style[i])
        # get datasets
        # train_path = os.path.join('../dataset/', 'train')
        # test_path = os.path.join('../dataset/s', 'test')
        # train_datasets, train_loader, test_datasets, test_loader = get_datasets(train_path, test_path)
        
        # get datasets        
        train_transforms_option = transforms.Compose([
                        transforms.Resize((32, 32)),
                        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
                    ])
        train_datasets = torchvision.datasets.SVHN('../dataset/svhn', split='train', download=True, transform = train_transforms_option)
        train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=256, shuffle=True,num_workers=8)
        
        test_transforms_option = transforms.Compose([
                        transforms.Resize((32, 32)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
                    ])
        test_datasets = torchvision.datasets.SVHN('../dataset/svhn', split='test', download=True, transform = test_transforms_option)
        test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=256, shuffle=True,num_workers=8)

        # model
        for model_path_nm in os.listdir('./checkpoint_0304/') :
            
            if 'svhn' in model_path_nm :
                print(model_path_nm) 

                model = resnet50(pretrained = False)
                num_ftrs = model.fc.in_features
                model.fc = nn.Linear(num_ftrs, 10) # num_classes
                model.load_state_dict(torch.load(os.path.join('./checkpoint_0304/', model_path_nm)))

                # optimizer
                if opt.adam:
                    optimizer = torch.optim.Adam(model.parameters(),lr=opt.learning_rate,weight_decay=0.0005)
                else:
                    optimizer = optim.SGD(model.parameters(),lr=opt.learning_rate,momentum=opt.momentum,weight_decay=opt.weight_decay)

                criterion = nn.CrossEntropyLoss()

                if torch.cuda.is_available():
                    if opt.n_gpu > 1:
                        model = nn.DataParallel(model)
                    model = model.cuda()
                    criterion = criterion.cuda()
                    cudnn.benchmark = True

                # set cosine annealing scheduler
                if opt.cosine:
                    eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.epochs, eta_min, -1)

                # start test
                print("==> Start eval...")
                test_acc, test_loss, test_f1 = test(test_loader, model, criterion, opt)
                print(f'test_acc : {test_acc}')
                print(f'test_loss : {test_loss}')
                print(f'test_f1 : {test_f1}')

                print()
            

        
        
        
if __name__ == '__main__' : 
    main()
    