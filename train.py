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
from util import adjust_learning_rate, accuracy, AverageMeter
from torchvision import transforms
from torchvision.models import resnet50
from sklearn.metrics import f1_score

def train(epoch, train_loader, model, criterion, optimizer, opt):
    """One epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc1 = AverageMeter()
    f1_1 = AverageMeter()
    
    total = 0
    correct = 0

    end = time.time()
    for idx, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        # ===================forward=====================
        output = model(input)
        loss = criterion(output, target)
        acc = accuracy(output, target)
        pred = output.argmax(dim=1) # .view(output.shape)

        f1 = f1_score(target.cpu().detach().numpy(), pred.cpu().detach().numpy(), average = 'weighted')
        losses.update(loss.item(), input.size(0))
        acc1.update(acc.item(), input.size(0))
        f1_1.update(f1.item(), input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # tensorboard logger
        pass

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'F1@1 {f1.val: 3f} ({f1.avg:3f})\t'.format(
                   epoch, idx, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=acc1, f1 = f1_1))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} F1@1 {f1.avg:.3f}'.format(top1=acc1, f1=f1_1))

    return acc1.avg, losses.avg



def validate(val_loader, model, criterion, opt):
    """One epoch validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc1 = AverageMeter()
    f1_1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (input, target) in enumerate(val_loader):

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)
            pred = output.argmax(dim=1) # .view(output.shape)
            f1 = f1_score(target.cpu().detach().numpy(), pred.cpu().detach().numpy(), average = 'weighted')

            # measure accuracy and record loss
            acc = accuracy(output, target)
            losses.update(loss.item(), input.size(0))
            acc1.update(acc, input.size(0))
            f1_1.update(f1, input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'F1@1 {f1.val:.3f} ({f1.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=acc1, f1=f1_1))

        print(' * Acc@1 {top1.avg:.3f}'
              .format(top1=acc1))

    return acc1.avg, losses.avg