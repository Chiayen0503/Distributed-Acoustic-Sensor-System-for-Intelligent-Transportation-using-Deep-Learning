import pandas as pd
import numpy as np
import os
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import random

def load_files(data_dir, files):
    data_list = [pd.read_csv(os.path.join(data_dir, f), delimiter=' ', header=None) for f in files]
    data = pd.concat(data_list, ignore_index=True)
    return data

def split_train_test(df, frac=.8, index=None):
    train = df.sample(frac=frac) if index is None else df.iloc[index]
    test = df.drop(train.index)
    return train, test, train.index

def split_train_test_2d(data, label, test_frac):
    seq = [i for i in range(data.shape[0])]
    n = round(data.shape[0]*test_frac)+1
    random_test_list = random.sample(seq, n)
    for i in random_test_list:
        seq.remove(i)
    x_train, y_train = data[seq], label[seq]
    x_test, y_test = data[random_test_list], label[random_test_list]
    return x_train, x_test, y_train, y_test, seq #train index 


def split_train_test_2d_balance(x, y, test_frac):

    arrays,counts = np.unique(y, axis = 0, return_counts = True)
    dominate_num = max(counts)
    
    #oversampling
    for a in arrays:
        result = np.where(y == a)
        i, c = np.unique(result[0],return_counts = True)
        data_num = len(i)
        
        i = i.tolist()

        if data_num < dominate_num:
            n_num = dominate_num - data_num
            random_oversampling_list = random.sample(i, n_num)

            random_oversample_y = y[random_oversampling_list]
            random_oversample_x = x[random_oversampling_list]
            y = np.concatenate((y, random_oversample_y), axis=0)
            x = np.concatenate((x, random_oversample_x), axis=0)
    
    
    seq = [i for i in range(x.shape[0])]
    n = round(x.shape[0]*test_frac)+1
    random_test_list = random.sample(seq, n)
    for i in random_test_list:
        seq.remove(i)
    x_train, y_train = x[seq], y[seq]
    x_test, y_test = x[random_test_list], y[random_test_list]
    return x_train, x_test, y_train, y_test, seq #train index 

def to_tensor(df):
    return np.array(df).reshape(df.shape[0], df.shape[1], 1)

def load_train_test_data(x_dir, x_files, y_dir, y_files, frac=.8):
    x = load_files(x_dir, x_files)
    y = load_files(y_dir, y_files)
    train_x, test_x = split_train_test(x, frac=frac)
    train_y, test_y = split_train_test(y, index=train_x.index)
    y0 = min(train_y[0].unique())
    train_y, test_y = tf.keras.utils.to_categorical(train_y - y0), tf.keras.utils.to_categorical(test_y - y0)
    return (to_tensor(train_x), train_y), (to_tensor(test_x), test_y)


def load_train_test_data_v2(x_dir, x_files, y_dir, y_files, frac=.8):
    x = load_files(x_dir, x_files)
    y = load_files(y_dir, y_files)
    train_x, test_x = split_train_test(x, frac=frac)
    train_y, test_y = split_train_test(y, index=train_x.index)
    y0 = min(train_y[0].unique())
    train_y, test_y = tf.keras.utils.to_categorical(train_y - y0), tf.keras.utils.to_categorical(test_y - y0)
    return (to_tensor(train_x), train_y), (to_tensor(test_x), test_y), train_x.index #return index to trace back loss func



def load_train_test_data_v3_balance(x_dir, x_files, y_dir, y_files, frac=.8):
    x = load_files(x_dir, x_files)
    y = load_files(y_dir, y_files)
    ground_truth_unique = y[0].unique()
    li = [y[0].value_counts()]
    dominate_num = max(li[0])
    
    #oversampling
    for l in ground_truth_unique:
        data_num = len(y[y[0]==l])
        if data_num < dominate_num:
            n_num = dominate_num - data_num
            random_sample_y = y[y[0]==l].sample(n=n_num)
            random_sample_x = x.iloc[random_sample_y.index]
            x = x.append(random_sample_x, ignore_index=True)
            y= y.append(random_sample_y, ignore_index=True)
      
    
    train_x, test_x = split_train_test(x, frac=frac)
    train_y, test_y = split_train_test(y, index=train_x.index)
    y0 = min(train_y[0].unique())
    train_y, test_y = tf.keras.utils.to_categorical(train_y - y0), tf.keras.utils.to_categorical(test_y - y0)
    return (to_tensor(train_x), train_y), (to_tensor(test_x), test_y), train_x.index #return index to trace back loss func



def train_test_data_recovery(x_dir, x_files, y_dir, y_files, train_index):
    x = load_files(x_dir, x_files)
    y = load_files(y_dir, y_files)
    train_x, test_x = split_train_test(x, index=train_index)
    train_y, test_y = split_train_test(y, index=train_index)
    y0 = min(train_y[0].unique())
    train_y, test_y = tf.keras.utils.to_categorical(train_y - y0), tf.keras.utils.to_categorical(test_y - y0)
    return (to_tensor(train_x), train_y), (to_tensor(test_x), test_y)


def extract_data(raw_label, raw_data, label): # extract data given specific label info
    df = pd.read_csv(raw_label, delimiter=' ', header = None, engine='python' )
    df2 = pd.read_csv(raw_data, delimiter=' ', header = None, engine='python' )
    sample_index = df.index[df.iloc[:, 0] == label].tolist()
    new_label = df.loc[sample_index]
    new_data = df2.loc[sample_index]
    print("output y shape: {0}".format(new_label.shape))
    print("output x shape: {0}".format(new_data.shape))
    return new_label, new_data