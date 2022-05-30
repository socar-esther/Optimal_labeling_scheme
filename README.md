# Shape prior is Not All You Need: Discovering Balance between Texture and Shape bias in CNN
Implementation of the paper submitted to NIPS 2022: "Shape prior is Not All You Need: Discovering Balance between Texture and Shape bias in CNN"

## Paper Abstract
As Convolutional NeuralNetwork (CNN) is known to be biased in image texture rather than object shapes trained under ImageNet dataset, recent works proposed that elevating shape awareness of the CNNs makes them similar to human visual recognition. However, beyond the ImageNet dataset, are CNNs biased to texture when they are fine-tuned on downstream datasets? How severely do CNNs have texture bias in the fine-tuning regime? If an ideal state of CNN resembles human perceptions, what solution can shorten the gap between current CNNs and human vision? In this paper, we present a series of analyses to establish answers to these questions. First, we propose AdaBA, a novel method of quantitatively illustrating CNN's shape and texture bias by resolving several limits of the previously-proposed method. Utilizing the proposed AdaBA, we discovered that fine-tuned CNNs also expose texture bias, but their strengths differ at the downstream dataset. Furthermore, we presume that previous solutions for enhancing CNN's shape awareness do not touch on the root cause of texture bias: data distribution. To this end, We propose a granular labeling scheme, a simple but effective solution for mitigating CNN's texture preference. Redesigning the label space with the proposed method, we empirically examine how the granular labeling scheme escalates classification accuracy and performance of OOD detection. We expect key findings and proposed methods in the study to elevate understanding of the CNN and yield an effective solution to mitigate this texture bias.

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



## Model checkpoints
- Experimental checkpoints of labeling scheme : [URL](https://drive.google.com/file/d/1yOz01OVvRCMu8DO_aqDgHNcLuHJfeJUC/view?usp=sharing)
- Transferred downstream weights : [URL](https://drive.google.com/file/d/1qvzIUqMkMfTd5_5wPMOFuRvgTRtPhj2G/view?usp=sharing)




