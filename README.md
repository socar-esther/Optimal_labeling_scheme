# Towards a Qualified Representation: Does a Good Vision Dataset Require Both Shape and Texture Bias Simultaneously?

## Paper Abstract
While the state-of-the-art deep neural networks utilized precisely-designed benchmark datasets, the real-world machine learning practitioners experience a challenge at the very first start: defining a good label space in their domain's problem. We conventionally say that a good dataset is necessary for an effective CNN; then, what's the definition of a good vision dataset? To pursue practical takeaways regarding this question, our study proposes a series of analyses regarding the definition of a good dataset. First, we set the CNN's performance in various vision tasks (target classification, transfer learning, and OOD detection) as a proxy of a good dataset's benefit. Second, we proposed a novel labeling scheme denoted as a granular scheme, where the samples at each label share both object shape and texture properties simultaneously. Throughout experimental studies, we discovered the following takeaways. First, the granular labeling scheme is superior to the other schemes for the target classification performance as it helps the model learn more various, fruitful representations on a given sample. Conversely, we figured out the granular labeling scheme is not always supreme in transferability and OOD detection, and this happens because the learned representation under the granular scheme is overly optimized to training samples without generalizability. By resolving the proposed improvement avenues, we expect academia to clarify the key properties of good vision datasets that can realize the benefits of deep neural networks in the real world shortly.

## How to Run
```python
## Baseline code
$ python main.py --learning_rate 0.01 \
                 --train_root [TRAIN_DATASET_ROOT] \ 
                 --test_root [TEST_DATA_ROOT] \ 
                 --n_class 10 \
                 --n_style [STYLE_NUMBER] \
                 --lr_decay_epochs '60,80,100,120,140,160,180'

## Transfer code
$ python transferability_main.py --learning_rate 0.01 \
                                 --train_root [TRAIN_DATASET_ROOT] \ 
                                 --test_root [TEST_DATA_ROOT] \ 
                                 --n_class [NUM_CLASS] \
                                 --n_style [STYLE_NUMBER] \
                                 --lr_decay_epochs '60,80,100,120,140,160,180'
```
- OOD detection code : [./notebook/ood.ipynb](https://github.com/socar-esther/cvprw_Optimal_labeling_scheme/blob/main/notebook/ood.ipynb)


## Model checkpoints
```python
## labeling scheme and ood weights
$ gsutil cp gs://socar-data-temp/esther/labeling_scheme/checkpoint_0302.zip . 

## transfered downstream weights
$ gsutil cp gs://socar-data-temp/esther/labeling_scheme/checkpoint_0317.zip .
```

## TODO
- [x] Check OOD detection notebook
- [x] Add model checkpoint 



