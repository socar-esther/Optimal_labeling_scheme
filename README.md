# Towards a Qualified Representation: Does a Good Vision Dataset Require Both Shape and Texture Bias Simultaneously?

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



