import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

trainset = pd.read_csv('trainset.csv').drop_duplicates()
trainset.label.replace(-1,0,inplace=True)
valset = pd.read_csv('valset.csv').drop_duplicates()
valset.label.replace(-1,0,inplace=True)

train_y = trainset.label
train_x = trainset.drop(['user_id','label','day_gap_before','day_gap_after'],axis=1)
val_y = valset.label
val_x = valset.drop(['user_id','label','day_gap_before','day_gap_after'],axis=1)

scaler = MinMaxScaler()
for col in train_x.columns:
    max_value = train_x[col].max()
    min_value = train_x[col].min()
    train_x[col] = (train_x[col]-min_value)/(max_value-min_value)
    val_x[col] = (val_x[col]-min_value)/(max_value-min_value)


def data_with_missing_value(batch_size=256):
    def trainset_generator():
        for i in range(train_x.shape[0]//batch_size):
            yield train_x[i*batch_size:(i+1)*batch_size],train_y[i*batch_size:(i+1)*batch_size].reshape((-1,1))

    return trainset_generator,(val_x,val_y.reshape((-1,1)))


def data_fill_zero(batch_size=256):
    train_x.fillna(0.0,inplace=True)
    val_x.fillna(0.0,inplace=True)
    def trainset_generator():
        for i in range(train_x.shape[0]//batch_size):
            yield train_x[i*batch_size:(i+1)*batch_size],train_y[i*batch_size:(i+1)*batch_size].reshape((-1,1))

    return trainset_generator,(val_x,val_y.reshape((-1,1)))


