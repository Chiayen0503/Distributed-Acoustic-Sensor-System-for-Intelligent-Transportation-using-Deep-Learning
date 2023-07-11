import numpy as np
from das_lib.funs import train_test_data_recovery

import pandas as pd
import numpy as np
import os
import tensorflow as tf
from sklearn.metrics import confusion_matrix

import json

def wrongly_predicted_indice(model, x_ind, y_ind):
    pred = model.predict(x_ind)
    pred = np.round(pred).astype(int)
    indices = [i for i,v in enumerate(pred) if not np.array_equal(pred[i], y_ind[i])]
    return indices


def save_wrong_indices(f_name, model, x_ind, y_ind):
    wrong_indice = wrongly_predicted_indice(model, x_ind, y_ind)
    wrong_dict = {'wrong_prediction_indices':wrong_indice}
    with open(f_name, "w") as outfile:
        json.dump(wrong_dict, outfile)
        
def save_training_history(f_name, H):
    performance_dict = {'train_acc':H.history['accuracy'], 'val_acc':H.history['val_accuracy'], 'acc_loss':H.history['loss'] ,'val_loss':H.history['val_loss']}    
    with open(f_name, "w") as outfile:
        json.dump(performance_dict, outfile)      


def rowdatagenerator(batch_size, data, labels):
    D = np.zeros((batch_size, data.shape[1]))
    l = np.zeros(batch_size)
    batch_idx = 0
    while True:
        for idx in np.random.permutation(data.index):
            D[batch_idx, :] = data.iloc[idx, :]
            l[batch_idx] = labels.iloc[idx]
            batch_idx += 1
            if batch_idx == batch_size:
                yield D, l
                batch_idx = 0
                
                
def get_confusion_matrix(model, testX, testy, class_num, ground_truth_unique_labels):
  cars = []
  if len(np.squeeze(testX).shape)<2: #1d
      for car in testX:
        car = car.reshape(1, testX.shape[1], 1)
        car = model.predict(car)
        cars.append(car)
  else:
      for car in testX: #2d
        car = car.reshape(1, testX.shape[1], testX.shape[2] , 1)
        car = model.predict(car)
        cars.append(car)
        
  cars = np.array(cars) 
  cars = cars.reshape(testX.shape[0], class_num)
  
#   try:
#     cars = cars.reshape(testX.shape[0], class_num)
#   except:
#     cars = cars.reshape(testX.shape[0], class_num-1)
  
  cars = np.argmax(cars, axis=1)
  cars = cars+1

  testy_copy = np.argmax(np.array(testy), axis=1)+1
  testy_copy = testy_copy.reshape(testX.shape[0],)
  test_c_m = confusion_matrix(y_true=testy_copy, y_pred=cars, labels = ground_truth_unique_labels )
  test_c_m = test_c_m.astype('float') / test_c_m.sum(axis=1)[:, np.newaxis]

  return test_c_m, testy_copy[0]




def print_average_cm(repeats,  models_root_folder, training_indices,class_num, data_dir, data_files, labels_dir, labels_files, ground_truth_unique_labels ):
    ten_conf_matrixs = []
    for i in range(repeats):
        model_folder = models_root_folder + "model_{0}".format(i+1)
        training_index = training_indices.iloc[i, 1:]
        (x_train_new, y_train_new), (x_test_new, y_test_new)  = train_test_data_recovery(data_dir, data_files, labels_dir, labels_files, training_index)
        reconstructed_model = tf.keras.models.load_model(model_folder)
        cm, sample_label = get_confusion_matrix(reconstructed_model, x_test_new, y_test_new, class_num, ground_truth_unique_labels)
        ten_conf_matrixs.append(cm)
    print('testing label: {0}'.format(sample_label))
    averaged_cm = sum(ten_conf_matrixs)/10
    print("average cm: {0}".format(averaged_cm))
    return averaged_cm

def print_average_cm_unseen(repeats,  models_root_folder, class_num, unseen_data, unseen_label, ground_truth_unique_labels):
    ten_conf_matrixs = []
    if unseen_label.shape[1]<2:
        y0 = 1
        unseen_label = tf.keras.utils.to_categorical(unseen_label - y0, num_classes=class_num)
    
    for i in range(repeats):
        model_folder = models_root_folder + "model_{0}".format(i+1)
        reconstructed_model = tf.keras.models.load_model(model_folder)
        cm, sample_label = get_confusion_matrix(reconstructed_model, unseen_data, unseen_label, class_num, ground_truth_unique_labels)
        ten_conf_matrixs.append(cm)
    print('testing label: {0}'.format(sample_label))
    averaged_cm = sum(ten_conf_matrixs)/10
    print("average cm: {0}".format(averaged_cm))
#     return averaged_cm
    return averaged_cm








