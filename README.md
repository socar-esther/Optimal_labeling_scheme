# Discovering the definition of good dataset

## How to Run
```python
$ python train.py --learning_rate 0.01 \
                  --train_root [TRAIN_DATASET_ROOT] \ 
                  --test_root [TEST_DATA_ROOT] \ 
                  --n_class 10 \
                  --n_style [STYLE_NUMBER] \
                  --lr_decay_epochs '60,80,100,120,140,160,180'
```
- OOD detection code : https://github.com/socar-esther/cvprw_Optimal_labeling_scheme/blob/main/notebook/ood.ipynb



## TODO
- [ ] Check OOD detection notebook (commit new version of code, this version is scratch)
- [ ] Add model checkpoint 


