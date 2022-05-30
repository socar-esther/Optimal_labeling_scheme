import models.resnet
from datasets.stylized_cifar10 import *
from torch.utils.data import DataLoader

from timm.models import create_model, apply_test_time_pool, load_checkpoint, is_model, list_models
from timm.data import create_dataset, create_loader, resolve_data_config, RealLabelsImagenet
from timm.utils import accuracy, AverageMeter, natural_key, setup_default_logging, set_jit_legacy


## Get dataset, model from other side of DB
def get_model(args) :
    
    if args.model == 'resnet50' : 
        model = models.resnet.resnet50(pretrained = args.pretrained)
    else :
        raise NotImplementedError('[INFO] check the implementation of the model')
        
    return model 


def get_dataloader(args, shape_pair_list, texture_pair_list) :
    dataset = CIFAR10(args, shape_pair_list, texture_pair_list)
    dataloader = DataLoader(dataset, args.batch_size, shuffle = False, num_workers = args.num_workers, pin_memory = True)
    return dataloader


## Estimate the dimention (for output shape)

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
    mean = np.mean(zall, 0, keepdim = True)
    var = np.sum(np.mean((zall-mean)*(zall-mean), 0))
    
    ## step 2-1. 각 factor별로 돌면서 다시 계산
    for f in range(args.n_factors) : 
        if f != arg.residual_index : 
            indices = np.where(factors == f)[0]
            za_by_factor[f] = za[indices]
            zb_by_factor[f] = zb[indices]
            mean_by_factors[f] = 0.5*(np.mean(za_by_factor[f], 0, keepdims=True)+np.mean(zb_by_factor[f], 0, keepdims=True))
            
            score_by_factor[f] = np.sum(np.mean((za_by_factor[f]-mean_by_factor[f])*(zb_by_factor[f]-mean_by_factor[f]), 0))
            score_by_factor[f] = score_by_factor[f]/var
            
            idv = np.mean((za_by_factor[f]-mean_by_factor[f])*(zb_by_factor[f]-mean_by_factor[f]), 0)/var
            individual_scores[f] = idv
        
        else : 
            # residual factor에 해당되는 경우
            score_by_factor[f] = 1.0
    
    scores = np.array([score_by_factor[f] for f in range(args.n_factors) ])
    
    # step3. softmax output (shape(0) or texture(1))
    m = np.max(scores)
    e = np.exp(scores-m)
    softmaxed = e/np.sum(e)
    
    dim = za.shape[1]
    dims = [int(s*dim) for s in softmaxed]
    
    dims[-1] = dim - sum(dims[:-1])
    dims_present = dims.copy()
    
    for i in range(len(dims)) : 
        dims_percent[i] = round(100*(dims[i] / sum(dims,)), 1)
    return dims, dims_percent