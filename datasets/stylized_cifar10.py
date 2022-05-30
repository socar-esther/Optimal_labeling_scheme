# 어차피 어떻게 들어가든 example1, example2의 형태로 나오면됨 (data agnostic으로 확인)
import os
import glob
import random
import cv2
import numpy as np
import torch.utils.data as data
import torch


class CIFAR10(data.Dataset) :
    
    def __init__(self, config, shape_pair_list, texture_pair_list) :
        
        self.shape_pair = shape_pair_list
        self.texture_pair = texture_pair_list
        
        # loader configuration
        self.n_factors = config.n_factors
        self.image_size = config.image_size
        self.prng = np.random.RandomState(1) # for random factor
        
    def get_random_factor(self, i) : 
        # shape(0) or Texture(1)를 선택하는 부분
        factor = self.prng.choice(2) 
        return factor
    
    def get_image(self, path) : 
        
        # open image
        image = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32)
        
        # resize
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation = cv2.INTER_LINEAR)
        image = self._normalize(image, mean=[0.456, 0.486, 0.485], std = [0.224, 0.225, 0.229])
        image = np.array(image)
        torch.tensor(image).permute(2, 0, 1)
        
        return torch.tensor(image).permute(2, 0, 1)
    
    def _normalize(self, image, mean = (0., 0., 0.), std = (1., 1., 1.)): 
        if mean[0] < 1: 
            image /= 255.0
        image -= mean
        image /= std
        return image
    
    def __getitem__(self, i) : 
        factor = random.randint(0, self.n_factors)
        
        # select random factor (0; shape, 1; texture)
        if factor == 0 :
            # same shape different texture
            example1 = self.get_image(self.shape_pair[i][0])
            example2 = self.get_image(self.shape_pair[i][1])
            
        else: 
            # same texture difference shape
            example1 = self.get_image(self.texture_pair[i][0])
            example2 = self.get_image(self.texture_pair[i][1])
        # else : 
        #     raise NotImplementedError('check the factor number (neither 0 nor 1)')
            
        return factor, example1, example2
    
    def __len__(self) :
        if len(self.shape_pair) > len(self.texture_pair) : 
            return len(self.texture_pair) 
        else : 
            return len(self.shape_pair)
        
        