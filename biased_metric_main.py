import os 
import copy
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

import config
from utils import *
import csv
from argparse import ArgumentParser
import json
from dim_utils import *
import warnings
warnings.filterwarnings(action='ignore')
device = "cuda:0"


def get_parser() : 
    
    # get configuration from argument
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--model_path', type=str, default='scratch/checkpoint/cifar10_best.pt', help='path to save model')
    parser.add_argument('--data_root', type=str, default='../dataset/stylized_cifar10_set/test_reduced_10', help='path to data root')
    parser.add_argument('--get_cluster', type=bool, default=False, help='check if we use del noise instances')
    opt = parser.parse_args()

    # load trained classifier
    opt.n_class = 10
    model = resnet50(pretrained = False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, opt.n_class)
    model.load_state_dict(torch.load(opt.model_path)) 
    
    for param in model.parameters():
        param.requires_grad_(False)
        
    model = model.cuda()
    
    return opt, model


def get_features(image, model):
    # get content, gram matrix from trained classifier
    x = image
    features = {}

    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    i = 0
    
    print(x.shape)

    for name, layer in model.layer1._modules.items():
        x = layer(x)
        i+=1
        features[f"bneck_{name}_{i}"] = x

    for name, layer in model.layer2._modules.items():
        x = layer(x)
        i+=1
        features[f"bneck_{name}_{i}"] = x

    for name, layer in model.layer3._modules.items():
        x = layer(x)
        i+=1
        features[f"bneck_{name}_{i}"] = x

    for name, layer in model.layer4._modules.items():
        x = layer(x)
        i+=1
        features[f"bneck_{name}_{i}"] = x

    return features     

def gram_matrix(tensor):
    # compute gram matrix from input feature
    _, d, h, w = tensor.shape
    tensor = tensor.reshape(d, h * w)
    gram = np.matmul(tensor, tensor.T)
    
    return gram

def get_dataset(opt) : 
    # get test dataset
    test_transforms_option = transforms.Compose([
                        transforms.Resize((256, 256)),
                        transforms.ToTensor(),
                        #transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
                    ])
    test_datasets = torchvision.datasets.ImageFolder(root=opt.data_root, transform = test_transforms_option)
    test_loader = torch.utils.data.DataLoader(test_datasets, batch_size = 16, shuffle=False, num_workers = 4)
    
    return test_datasets, test_loader


def run_main() :
    # get configuration, model
    opt, model = get_parser()
    
    # get dataset
    test_datasets, test_loader = get_dataset(opt)
    
    # save the features from trained classifiers' each layer
    i = 0
    image_features_output = dict()

    with torch.no_grad() :

        for idx, (input, target) in enumerate(test_loader) :
            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            image_features = get_features(input, model)

            if i == 0 :
                for key, item in image_features.items():
                    image_features_output[key] = item
            else :
                for key, item in image_features.items() : 
                    image_features_output[key] = torch.cat([image_features_output[key], image_features[key]])

            i += 1
    
    # get content matrix from higher layer (assigned 'bneck_4_12' in resnet50)
    content_mat = image_features_output['bneck_4_12'].cpu().detach().numpy()

    # get gram matrix from lower layer of trained classifier
    i = 0

    for j in range(content_mat.shape[0]) :

        gram_tmp_output = gram_matrix(image_features_output['bneck_1_2'][j].unsqueeze(0).cpu().detach().numpy())
        gram_tmp_output = np.expand_dims(gram_tmp_output, axis=0)

        if i == 0 :
            gram_output = gram_tmp_output
        else :
            gram_output = np.concatenate((gram_output, gram_tmp_output), axis = 0 )
        i += 1


    # reshape content matrix, gram matrix to 2-D shape
    gram_output_2d = np.reshape(gram_output, (gram_output.shape[0],(gram_output.shape[1] * gram_output.shape[1] )))
    content_output_2d = np.reshape(content_mat, (content_mat.shape[0], (content_mat.shape[1] * content_mat.shape[2] * content_mat.shape[3] )))

    print('-- gram 2-Dmatrix shape : ', gram_output_2d.shape)
    print('-- content matrix 2-D shape : ', content_output_2d.shape)

    # get filenames from image directory
    file_nm_list = list()

    for i in range(len(test_datasets.imgs)) : 
        file_nm = test_datasets.imgs[i][0]
        file_nm_list.append(file_nm)

    cluster_df = pd.DataFrame({'file_nm' : file_nm_list})

    outlier_texture_list = list()
    outlier_list = list()

    # filter the outlier (using clustering, optional)
    if opt.get_cluster : 
        # shape cluster
        print('Start shape clustering (10) ...')
        k = 10
        model = KMeans(n_clusters = k, random_state = 10, max_iter = 1000)
        model.fit(content_output_2d)

        cluster_df['shape_cluster'] = model.fit_predict(content_output_2d)
        print(cluster_df['shape_cluster'].value_counts())

        # TODO; adjust depends on datasets' characteristic
        for cluster_id in cluster_df['shape_cluster'].value_counts().index.tolist()[5:] : 
            outlier_list.extend(cluster_df[cluster_df['shape_cluster'] == cluster_id]['file_nm'].to_list())

        # texture cluster
        print('Start texture clustering (5) ...')
        k = 5
        model = KMeans(n_clusters = k, random_state = 10, max_iter = 1000)
        model.fit(gram_output_2d)

        cluster_df['texture_cluster'] = model.fit_predict(gram_output_2d)
        print(cluster_df['texture_cluster'].value_counts())

        # TODO; adjust depends on datasets' characteristic
        for cluster_id in cluster_df['texture_cluster'].value_counts().index.tolist()[4:] : 
            outlier_texture_list.extend(cluster_df[cluster_df['texture_cluster'] == cluster_id]['file_nm'].to_list())



    # (1) Get shape pair
    texture_pair_list = list()

    for i in range(len(cluster_df)):
        tmp_pair_list = list()

        # print(cluster_df['file_nm'][i])
        tmp_pair_list.append(cluster_df['file_nm'][i])
        content_mat_i = content_mat[i]
        gram_mat_i = gram_output[i]

        loss_list = list()

        ## get distance from anchor image
        for j in range(len(cluster_df)):

            content_mat_j = content_mat[j]
            gram_mat_j = gram_output[j]

            # shape, texture loss
            loss_shape = ( content_mat_i - content_mat_j ) ** 2
            loss_texture = ( gram_mat_i - gram_mat_j ) ** 2

            # normalize (texture scale >>> shape scale)
            normalized_shape = loss_shape / np.sqrt(np.sum(loss_shape))
            normalized_texture = loss_texture / np.sqrt(np.sum(loss_texture))

            if i == j :
                loss = 100000000000
            else : 
                loss = np.mean(normalized_texture) ** 2 / np.mean(normalized_shape)
            loss_list.append(loss)

        ## closest index
        min_idx = np.argmin(np.array(loss_list))

        shape_distance_min = cluster_df['file_nm'][min_idx]#[min_idx+(i+1)]
        tmp_pair_list.append(shape_distance_min)

        if len(tmp_pair_list) !=2 : 
            print('check the local lenghts of the list!')
            
        texture_pair_list.append(tmp_pair_list)

    print('- Lengths of texture pair list :', len(texture_pair_list))\
    
    ## (2) Get texture pair
    correct = 0
    shape_pair_list = list()


    for i in range(len(cluster_df)):
        if cluster_df['file_nm'][i] not in outlier_texture_list  : 

            tmp_pair_list = list()

            content_mat_i = content_mat[i]
            gram_mat_i = gram_output[i]

            loss_list = list()

            ## get distance with the anchor image
            for j in range(len(cluster_df)):

                if cluster_df['file_nm'][j] not in outlier_texture_list : 

                    content_mat_j = content_mat[j]
                    gram_mat_j = gram_output[j]

                    # shape, texture loss (l2 distance)
                    loss_shape = ( content_mat_i - content_mat_j ) ** 2
                    loss_texture = ( gram_mat_i - gram_mat_j ) ** 2

                    # normalize (texture scale >>> shape scale)
                    normalized_shape = loss_shape / np.sqrt(np.sum(loss_shape)) 
                    normalized_texture = loss_texture / np.sqrt(np.sum(loss_texture)) 

                    if i == j :
                        loss = 1000000000
                    else : 
                        loss = np.mean(normalized_shape) / np.mean(normalized_texture) # np.mean(normalized_shape) / np.mean(normalized_texture) 
                    loss_list.append(loss)

            ## get closest index image
            min_idx = np.argmin(np.array(loss_list))
            shape_distance_min = cluster_df['file_nm'][min_idx]#[min_idx+(i+1)]
            tmp_pair_list.append(cluster_df['file_nm'][i])

            tmp_pair_list.append(shape_distance_min)

            if len(tmp_pair_list) != 2 : 
                print('check the lenghts of the tmp pair list!')
                break

            shape_pair_list.append(tmp_pair_list)
    
    print('- Lengths of shape pair list : ', len(shape_pair_list))
            
    # get shape biased score (w/ pre-defined equations)
    args, model = get_dim_parser(opt)

    # get dataset
    dataloader = get_dataloader(args, shape_pair_list, texture_pair_list)

    # construct factor list
    factor_list = list()
    output_dict = {'example1' : list(), 
                   'example2' : list() }

    print('-- Start preprocessing to estimate dim')
    for i, (factor, example1, example2) in enumerate(dataloader) : 
        example1, example2 = example1.cuda(device), example2.cuda(device)

        # pass images through model and get distribution mean value
        output1 = model(example1).mode()[0]
        output2 = model(example2).mode()[0]

        # save factor and output vector
        factor_list.append(factor.detach().cpu().numpy() )
        output_dict['example1'].append(output1.detach().cpu().numpy() )
        output_dict['example2'].append(output2.detach().cpu().numpy() )

        if i % 10 == 0 : 
            print(f'-- check Iter {i}/{len(dataloader)}')


    dims, dims_percent = dim_est(output_dict, factor_list, args)

    print(" >>> Estimated factor dimensionalities: {}".format(dims))
    print(" >>> Ratio to total dimensions: {}".format(dims_percent))
    print('Saving results to {}'.format(args.save_dir))
    
    
    
if __name__ == '__main__' :
    run_main()