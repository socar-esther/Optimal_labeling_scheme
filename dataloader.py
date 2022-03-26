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
from util import adjust_learning_rate, accuracy, AverageMeter
from torchvision import transforms
from torchvision.models import resnet50
from sklearn.metrics import f1_score


def get_datasets(train_root, test_root) : 
    print('>> Training path : ', train_root)
    print('>> Test path : ', test_root)
    
    
    # get datasets
    train_transforms_option = transforms.Compose([
                        transforms.Resize((32, 32)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
                    ])
    train_datasets = torchvision.datasets.ImageFolder(root=train_root, transform = train_transforms_option)
    train_loader = torch.utils.data.DataLoader(train_datasets, batch_size = 256, shuffle=True, num_workers = 4)


    test_transforms_option = transforms.Compose([
                        transforms.Resize((32, 32)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
                    ])
    test_datasets = torchvision.datasets.ImageFolder(root=test_root, transform = test_transforms_option)
    test_loader = torch.utils.data.DataLoader(test_datasets, batch_size = 256, shuffle=False, num_workers = 4)
    
    return train_datasets, train_loader, test_datasets, test_loader
