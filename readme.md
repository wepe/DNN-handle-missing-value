## DNN handling missing value

- imputation with 0.0, refer to `nn.py`
- adative imputation, refer to `hmv_nn.py`. 

    if x is missing:
        w1
    else:
        w2*x

  w1 and w2 are trainable

- use the [O2O data](https://pan.baidu.com/s/1nvFG2ff), extract feature using [extract_feature.py](https://github.com/wepe/O2O-Coupon-Usage-Forecast/blob/master/code/wepon/season%20one/extract_feature.py)
