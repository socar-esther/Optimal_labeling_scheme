from __future__ import print_function

import os
import argparse
import socket
import time
import sys
from PIL import Image

import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset

import torchvision
from util import adjust_learning_rate, accuracy, AverageMeter
from torchvision import transforms
from torchvision.models import resnet50
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd
import numpy as np

import csv
import cv2

print(torch.__version__)

def parse_option() :
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--eval_freq', type=int, default=10, help='meta-eval frequency')
    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=10, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=4, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=500, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,80,100,120,150,180,200', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--adam', action='store_true', help='use adam optimizer')

    # dataset
    parser.add_argument('--model', type=str, default='resnet50')

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

def make_data_csv() :
    style_list = ['style4']
    
    for style_nm in style_list :
        train_path = f'../dataset/ours_{style_nm}_set/train'
        test_path = f'../dataset/ours_{style_nm}_set/test'
        result_dict = {
            "img_path" : [],

            "airplane" : [], 
            "automobile" : [],
            "bird" : [],
            "cat" : [],
            "deer" : [],
            "dog" : [],
            "frog" : [],
            "horse" : [],
            "ship": [],
            "truck" : [],
            "original" : [], 
            "stylized" : []
        }

        # 총 클래스는 shape(10개) + original + style1 ; 총 12개로 두면 될듯
        for class_nm in os.listdir(train_path) :
            if '.ipy' in class_nm :
                continue
            else :
                # 각 이미지 path를 설정
                for file_nm in os.listdir(os.path.join(train_path, class_nm) ) : 
                    img_path = os.path.join(train_path, class_nm, file_nm)

                    # file path 먼저 채우고
                    result_dict['img_path'].append(img_path)

                    # texture부분 채우고
                    if style_nm in class_nm :
                        result_dict['stylized'].append(1)
                        result_dict['original'].append(0)
                    else :
                        result_dict['original'].append(1)
                        result_dict['stylized'].append(0)

                    # shape부분 채운다
                    shape_list = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

                    for shape_nm in shape_list :
                        if shape_nm in class_nm :
                            result_dict[shape_nm].append(1)

                            # shape_nm 이외에 다른 shape들에는 0 값 넣음
                            for shape_nm_2 in shape_list :
                                if shape_nm_2 != shape_nm :
                                    result_dict[shape_nm_2].append(0)
                            break
    
        result_test_dict = {
            "img_path" : [],

            "airplane" : [], 
            "automobile" : [],
            "bird" : [],
            "cat" : [],
            "deer" : [],
            "dog" : [],
            "frog" : [],
            "horse" : [],
            "ship": [],
            "truck" : [],
            "original" : [], 
            "stylized" : []
        }


        for class_nm in os.listdir(test_path) :
            if '.ipy' in class_nm :
                continue
            else :
                # 각 이미지 path를 설정
                for file_nm in os.listdir(os.path.join(test_path, class_nm) ) : 
                    img_path = os.path.join(test_path, class_nm, file_nm)

                    # file path 먼저 채우고
                    result_test_dict['img_path'].append(img_path)

                    # texture부분 채우고
                    if style_nm in class_nm :
                        result_test_dict['stylized'].append(1)
                        result_test_dict['original'].append(0)
                    else :
                        result_test_dict['original'].append(1)
                        result_test_dict['stylized'].append(0)

                    # shape부분 채운다
                    shape_list = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

                    for shape_nm in shape_list :
                        if shape_nm in class_nm :
                            result_test_dict[shape_nm].append(1)

                            # shape_nm 이외에 다른 shape들에는 0 값 넣음
                            for shape_nm_2 in shape_list :
                                if shape_nm_2 != shape_nm :
                                    result_test_dict[shape_nm_2].append(0)
                            break
        train_csv = pd.DataFrame(result_dict)
        test_csv = pd.DataFrame(result_test_dict)
        
        # csv 파잎 저장
        train_csv.to_csv(f'../dataset/ours_{style_nm}_set/train.csv', index = False)
        test_csv.to_csv(f'../dataset/ours_{style_nm}_set/test.csv', index = False)
        
    


class MultiLabel_Styleized_CIFAR10(Dataset):
    def __init__(self, image_ids, transforms) :
        self.transforms = transforms

        self.labels = {}
        with open(image_ids, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                self.labels[row[0]] = list(map(int, row[1:]))

        self.image_ids = list(self.labels.keys())

    def __len__(self) :
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        image = Image.open(
            os.path.join(f'{str(image_id)}')).convert('RGB')
        target = np.array(self.labels.get(image_id)).astype(np.float32)

        if self.transforms is not None:
            image = self.transforms(image)

        return image, target
    
    
def train(epoch, train_loader, model, criterion, optimizer, opt):
    """One epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    total = 0
    correct = 0
    model.train()
    
    end = time.time()
    for idx, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        # ===================forward=====================
        output = model(input)
        output = torch.sigmoid(output)
        loss = criterion(output, target)
        
        losses.update(loss.item(), input.size(0))

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
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch, idx, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()

    return losses.avg


def accuracy_for_test(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    correct_count = 0.0
    
    for i in range(len(targets)) :
        if predictions[i] == targets[i] :
            correct_count += 1.0
            
    return correct_count / len(targets)

    
def validate(val_loader, model, criterion, opt):
    """One epoch validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()

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
            output = torch.sigmoid(output)
            loss = criterion(output, target)

            losses.update(loss.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                       idx, len(val_loader), batch_time=batch_time, loss=losses,))


    return losses.avg

    

def main() :
    opt = parse_option()
    make_data_csv()
    
    ## dataset
    train_transforms_option = transforms.Compose([
                        transforms.Resize((32, 32)),
                        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
                    ])

    test_transforms_option = transforms.Compose([
                        transforms.Resize((32, 32)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
                    ])
    
    style_list = ['style4']
    
    for style_nm in style_list :
        
        print(f'Start with {style_nm} !!')
        trainset = MultiLabel_Styleized_CIFAR10(f'../dataset/ours_{style_nm}_set/train.csv', train_transforms_option)
        testset = MultiLabel_Styleized_CIFAR10(f'../dataset/ours_{style_nm}_set/test.csv', test_transforms_option)

        train_loader = DataLoader(trainset, batch_size=256, num_workers=8)
        test_loader = DataLoader(testset, batch_size=32, num_workers=4)
        
        # model load
        print(f'We use {opt.model}')
        model = resnet50(pretrained = False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 12)
        
        # optimizer
        if opt.adam:
            optimizer = torch.optim.Adam(model.parameters(),lr=opt.learning_rate,weight_decay=0.0005)
        else:
            optimizer = optim.SGD(model.parameters(),lr=opt.learning_rate,momentum=opt.momentum,weight_decay=opt.weight_decay)

        criterion = nn.MultiLabelSoftMarginLoss() #nn.BCELoss()

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
            
        
        # train 시작
        best_acc = 0.0
        best_loss = 1000.0

        for epoch in range(1, opt.epochs + 1):
            if opt.cosine:
                scheduler.step()
            else:
                adjust_learning_rate(epoch, opt, optimizer)
            print("==> training...")

            time1 = time.time()
            train_loss = train(epoch, train_loader, model, criterion, optimizer, opt)
            time2 = time.time()
            print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

            print(f'[Epoch{epoch}] train_loss', train_loss)

            test_loss = validate(test_loader, model, criterion, opt)

            print(f'[epoch {epoch}] test_loss : {test_loss}')

            # regular saving
            if best_loss > test_loss :
                best_loss = test_loss
                print('==> Saving...')
                save_file = os.path.join(f'./checkpoint_multi/ours_{style_nm}_best.pth')
                torch.save(model.state_dict(), save_file)
                
                
                # 여기서 test 한번 날려주기
                model= model.cuda()
                model.eval()

                losses = AverageMeter()
                acc1 = AverageMeter()
                f1_1 = AverageMeter()

                with torch.no_grad() : 
                    for idx, (input, target) in enumerate(test_loader):
                        input = input.cuda()
                        target = target.cuda()

                        target_list = list()
                        for i in range(len(target)) :
                            target_list.append(target[i].cpu().detach().numpy()[:10])

                        target_arr = torch.tensor(np.array(target_list))
                        


                        # get all the index positions where value == 1
                        target_indices = [i for i in range(len(target_arr[0])) if target_arr[0][i] == 1]

                        # get the predictions by passing the image through the model
                        outputs = model(input)
                        outputs_list = list()
                        for i in range(len(outputs)) :
                            outputs_list.append(outputs[i].cpu().detach().numpy()[:10])

                        outputs_arr = np.array(outputs_list)

                        outputs = torch.sigmoid(torch.tensor(outputs_arr))
                        outputs = outputs.detach().cpu()
                        
                        pred = outputs.argmax(dim=1)
                        targets = target_arr.argmax(dim=1)


                        #f1 = f1_score(target_arr.cpu().detach().numpy(), pred.cpu().detach().numpy(), average = 'weighted')
                        acc = accuracy_for_test(outputs, targets.cpu().detach().numpy())

                        acc1.update(acc, input.size(0))
                        #f1_1.update(f1, input.size(0))


                        if idx % opt.print_freq == 0:
                            print('Test: [{0}/{1}]\t'
                                    'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                                    #'F1@1 {f1.val:.3f} ({f1.avg:.3f})'
                                      .format(
                                       idx, len(test_loader),
                                       top1=acc1))#, f1=f1_1))

                print('Contexture (texture) Acc : ', acc1.avg)
                
                losses = AverageMeter()
                acc1 = AverageMeter()
                f1_1 = AverageMeter()

                with torch.no_grad() : 
                    for idx, (input, target) in enumerate(test_loader):
                        input = input.cuda()
                        target = target.cuda()

                        target_list = list()
                        for i in range(len(target)) :
                            target_list.append(target[i].cpu().detach().numpy()[-2:])

                        target_arr = torch.tensor(np.array(target_list))


                        # get all the index positions where value == 1
                        target_indices = [i for i in range(len(target_arr[0])) if target_arr[0][i] == 1]

                        # get the predictions by passing the image through the model
                        outputs = model(input)
                        outputs_list = list()
                        for i in range(len(outputs)) :
                            outputs_list.append(outputs[i].cpu().detach().numpy()[-2:])

                        outputs_arr = np.array(outputs_list)

                        outputs = torch.sigmoid(torch.tensor(outputs_arr))

                        outputs = outputs.detach().cpu()

                        pred = outputs.argmax(dim=1)
                        targets = target_arr.argmax(dim=1)


                        #f1 = f1_score(target_arr.cpu().detach().numpy(), pred.cpu().detach().numpy(), average = 'weighted')
                        acc = accuracy_for_test(outputs, targets.cpu().detach().numpy())

                        acc1.update(acc, input.size(0))
                        #f1_1.update(f1, input.size(0))


                        if idx % opt.print_freq == 0:
                            print('Test: [{0}/{1}]\t'
                                    'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                                    #'F1@1 {f1.val:.3f} ({f1.avg:.3f})'
                                      .format(
                                       idx, len(test_loader),
                                       top1=acc1))#, f1=f1_1))

                print('Conttexture (shape) Acc ', acc1.avg)
            
        
        
        
        
        
    
    
if __name__ == "__main__" :
    main()

    
