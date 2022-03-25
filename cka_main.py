# cka main
import os
from torch_cka import CKA
import torch
import torchvision
from torchvision.models import resnet50
from torchvision import transforms
import numpy as np

def get_dataset() :
    test_transforms_option = transforms.Compose([
                        transforms.Resize((32, 32)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
                    ])
    test_datasets = torchvision.datasets.ImageFolder(root="./dataset/test_small_10", transform = test_transforms_option)
    dataloader = torch.utils.data.DataLoader(test_datasets, batch_size = 16, shuffle=False, num_workers = 4)

    return test_datasets, dataloader

def save_matrix(file_name, matrix):
    np.savetxt(file_name, matrix, delimiter=",")
    
    return 0


if __name__ == '__main__' :
    
    # weight 전체 돌면서 값 저장하기
    weight_path = './checkpoint/'
    
    # 중복체크용 딕셔너리 지정
    duplicate_dict = dict()
    
    print('-- check all the weight files in checkpoint ')
    for file_nm in os.listdir(weight_path):
        print(file_nm)
        
    test_datasets, dataloader = get_dataset()

    for weight_nm_1 in os.listdir(weight_path) :
        weight_path_1 = os.path.join('./checkpoint', weight_nm_1)
        model1 = resnet50(pretrained=False)  
        model1.load_state_dict(torch.load(weight_path_1))
        
        # 레이어 추가
        layer_1_list = list()
        
        for name, module in model1.named_modules() :
            if 'conv' in name :
                layer_1_list.append(name)
                
        for weight_nm_2 in os.listdir(weight_path) :
            weight_path_2 = os.path.join('./checkpoint', weight_nm_2)
            model2 = resnet50(pretrained=False)
            model2.load_state_dict(torch.load(weight_path_2))
            
            # 레이어 추가
            layer_2_list = list()
            
            for name, module in model2.named_modules() :
                if 'conv' in name :
                    layer_2_list.append(name)
            
            
            # cka 모듈
            cka = CKA(model1, model2,
                     model1_name = str(weight_nm_1), 
                     model2_name = str(weight_nm_2),
                     model1_layers = layer_1_list,
                     model2_layers = layer_2_list,
                     device = 'cuda')
            
            # output으로 출력하고 싶은 것들 추가
            check_1 = str(model_nm_1) + '-' + str(model_nm_2)
            check_2 = str(model_nm_2) + '-' + str(model_nm_1)
            
            if check_1 in duplicate_dict.keys() or check_2 in duplicate_dict.keys() :
                continue
            else :
                cka.compare(dataloader)
                results = cka.export()
                save_path=f"./result/{str(model_nm_1)}-{str(model_nm_2)}_compare.png"
                cka.plot_results(save_path=save_path)

                duplicate_dict[str(model_nm_1) + '-' + str(model_nm_2)] = 1