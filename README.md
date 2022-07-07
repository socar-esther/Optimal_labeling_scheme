# Shape prior is Not All You Need: Discovering Balance between Texture and Shape bias in CNN
Implementation of the paper submitted to ACCV 2022: "Shape prior is Not All You Need: Discovering Balance between Texture and Shape bias in CNN"

## Paper Abstract
As Convolutional Neural Network (CNN) trained under ImageNet is known to be biased in image texture rather than object shapes, recent works proposed that elevating shape awareness of the CNNs makes them similar to human visual recognition. However, beyond the ImageNet-trained CNN, how can we make CNNs similar to human vision in the wild? In this paper, we present a series of analyses to answer this question. First, we propose AdaBA, a novel method of quantitatively illustrating CNN's shape and texture bias by resolving several limits of the prior method. With the proposed AdaBA, we focused on fine-tuned CNN's bias landscape which previous studies have not dealt with. We discover that fine-tuned CNNs are also biased to texture, but their bias strengths differ along with the downstream dataset; thus, we presume a data distribution is a root cause of texture bias exists. To tackle this root cause, we propose a granular labeling scheme, a simple but effective solution that redesigns the label space to pursue a balance between texture and shape biases. We empirically examine that the proposed scheme escalates CNN's classification and OOD detection performance. We expect key findings and proposed methods in the study to elevate understanding of the CNN and yield an effective solution to mitigate this texture bias.

## How to Run
### (1) AdaBA : bias measurement
- Here is the sample commands for running the AdaBA (ours)
- If you want to see the scratch notebook code, refer this: [URL](https://github.com/socar-esther/Optimal_labeling_scheme/blob/main/notebook/scracth_AdaBA.ipynb)
```python
## AdaBA script
$ python biased_metric_main.py --model resnet50 \ 
                               --model_path scratch/checkpoint/cifar10_best.pt \ 
                               --data_root ../dataset/stylized_cifar10_set/test_reduced_10
```

### (2) Granular Scheme Experimental Codes
- sample commands for running Granular Scheme experiment: (a) Classification task, (b) OOD Task, (c) Transfer Task
```python                            
## Granular Scheme Baseline code (for classification)
$ python main.py --learning_rate 0.01 \
                 --train_root [TRAIN_DATASET_ROOT] \ 
                 --test_root [TEST_DATA_ROOT] \ 
                 --n_class 10 \
                 --n_style [STYLE_NUMBER] \
                 --lr_decay_epochs '60,80,100,120,140,160,180'

## Granular Scheme Transferability code
$ python transferability_main.py --learning_rate 0.01 \
                                 --train_root [TRAIN_DATASET_ROOT] \ 
                                 --test_root [TEST_DATA_ROOT] \ 
                                 --n_class [NUM_CLASS] \
                                 --n_style [STYLE_NUMBER] \
                                 --lr_decay_epochs '60,80,100,120,140,160,180'
```

## Usage
**Stylized CIFAR-10 Preparation**
- Download the CIFAR-10 dataset from [official website](https://www.cs.toronto.edu/~kriz/cifar.html)
- Get style images (paintings). Download train.zip file from [Kaggles' painter-by-numbers Dataset](https://www.kaggle.com/c/painter-by-numbers/data)



## Model checkpoints
- Experimental checkpoints of labeling scheme : [URL](https://drive.google.com/file/d/1yOz01OVvRCMu8DO_aqDgHNcLuHJfeJUC/view?usp=sharing)
- Transferred downstream weights : [URL](https://drive.google.com/file/d/1qvzIUqMkMfTd5_5wPMOFuRvgTRtPhj2G/view?usp=sharing)




