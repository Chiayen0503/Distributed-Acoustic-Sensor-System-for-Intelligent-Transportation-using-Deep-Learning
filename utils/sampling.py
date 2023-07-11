import numpy as np
import pandas as pd
def extract_car_data(f_or_np_X, f_or_np_y, car_label): # extract data given specific label info
    
#     df1 = pd.read_csv(f_x, delimiter=' ', header = None)
#     df2 = pd.read_csv(f_y, delimiter=' ', header = None)
    df1, df2 = wrapper_to_pandas_df(f_or_np_X, f_or_np_y)
    sample_index = df2.index[df2.iloc[:, 0] == car_label].tolist()
    new_data = df1.loc[sample_index]
    return new_data

def wrapper_to_pandas_df(f_or_np_X, f_or_np_y):
    if type(f_or_np_X)!=type(f_or_np_y):
        raise SystemExit("f_or_npX and f_or_npy have to share the same type!")
    elif type(f_or_np_X) == str:
        df1 = pd.read_csv(f_or_np_X, delimiter=' ', header = None)
        df2 = pd.read_csv(f_or_np_y, delimiter=' ', header = None)
    elif type(f_or_np_X) == np.ndarray:
        df1 = pd.DataFrame(f_or_np_X)
        df2 = pd.DataFrame(f_or_np_y)
    elif type(f_or_np_X) == pd.core.frame.DataFrame:
        return f_or_np_X, f_or_np_y
    else:
        raise SystemExit("Only support .txt, np.array or pandas dataframe")
    return df1, df2
    

def generate_2d_data(data, car_label, bin_length, stride, total_bin):
    total_bin = data.shape[0] if total_bin is None else total_bin
    #window parameters
    head = 0
    tail = bin_length
    
    #save window
    two_d_data_list = []
    count_window = 0
    
    while tail < total_bin:
        two_d_data = np.array(data[head:tail][:])
        two_d_data_list.append(two_d_data)
        count_window+=1
        head+= stride
        tail+= stride
    return np.array(two_d_data_list), np.array([car_label]*count_window) 


def get_all_cars_2d(f_x, f_y, unique_labels, bin_length, stride, total_bin):
    aggregate_data = []
    aggregate_label = []
    for car_label in unique_labels:
        car_1d_data =  extract_car_data(f_x, f_y, car_label)
        car_2d_data, car_2d_label = generate_2d_data(car_1d_data, car_label, bin_length, stride, total_bin)
        aggregate_data.append(car_2d_data)
        aggregate_label.append(car_2d_label)
        
    return np.concatenate(aggregate_data,axis=0), np.concatenate(aggregate_label, axis=0)


def expand_data(data, label, num_classes):
    data= np.expand_dims(data,-1)
    label = np.array([label]*len(data))
    label = tf.keras.utils.to_categorical(label-1, num_classes)
    return data, label