import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


def get_test_data(df,p_scaler,rc_scaler,rt_scaler):
    p_data = np.array(df['total_purchase_amt'])
    rc_data = np.array(df['consume_amt'])
    rt_data = np.array(df['transfer_amt'])
    p_test = p_scaler.transform(p_data.reshape(p_data.shape[0],-1))
    rc_test = rc_scaler.transform(rc_data.reshape(rc_data.shape[0],-1))
    rt_test = rt_scaler.transform(rt_data.reshape(rc_data.shape[0],-1))
    return p_test,rc_test,rt_test

# def train_valid_split(data,split_index):
#     return data[:split_index], data[split_index:]

# def get_tensor_dataset(data,split_index,WINDOW_SIZE, BATCH_SIZE, SHUFFLE_BUFFER,number_pred,number_shift,y_col=0):
#     train,valid = train_valid_split(data,split_index)
#     tensor_train = ts_data_generator(train, WINDOW_SIZE, BATCH_SIZE, SHUFFLE_BUFFER,number_pred,number_shift,y_col)
#     tensor_valid = ts_data_generator(valid, WINDOW_SIZE, BATCH_SIZE, SHUFFLE_BUFFER,number_pred,number_shift,y_col)
#     return tensor_train,tensor_valid

def pr_scaling(df_poor):
    p_data = np.array(df_poor['total_purchase_amt'])
    rc_data = np.array(df_poor['consume_amt'])
    rt_data = np.array(df_poor['transfer_amt'])
    p_scaler = MinMaxScaler(feature_range=(0, 1))
    rc_scaler = MinMaxScaler(feature_range=(0, 1))
    rt_scaler = MinMaxScaler(feature_range=(0, 1))
    p_data = p_scaler.fit_transform(p_data.reshape(-1,1)).reshape(len(p_data),1)
    rc_data = rc_scaler.fit_transform(rc_data.reshape(-1,1)).reshape(len(rc_data),1)
    rt_data = rt_scaler.fit_transform(rt_data.reshape(-1,1)).reshape(len(rt_data),1)
    return p_data,rc_data,rt_data,p_scaler,rc_scaler,rt_scaler

def x_scaling(df):
    data = df.values
    x_scaler = MinMaxScaler(feature_range=(0, 1))
    x_data = x_scaler.fit_transform(data)
    return x_data,x_scaler

# def ts_data_generator(data,window_size, batch_size, shuffle_buffer, number_pred=30, shift=30,y_col = 0,dimension = 3):
#     ts_data = tf.data.Dataset.from_tensor_slices(data)
#     ts_data = ts_data.window(window_size + number_pred, shift=shift, drop_remainder=True)
#     ts_data = ts_data.flat_map(lambda window: window.batch(window_size + number_pred))
#     if dimension == 3:
#         ts_data = ts_data.shuffle(shuffle_buffer).map(
#             lambda window:(tf.reshape(window[:-number_pred], [window_size,data.shape[1]]),
#                            tf.reshape(window[-number_pred:,y_col], [number_pred, 1])))
#     else:
#         ts_data = ts_data.shuffle(shuffle_buffer).map(
#             lambda window:(tf.reshape(window[:-number_pred], [window_size,data.shape[1],1]),
#                            tf.reshape(window[-number_pred:,y_col], [number_pred, 1])))
#     ts_data = ts_data.batch(batch_size).prefetch(1)
#     return ts_data

class Processer(object):
    def __init__(self, window_size, batch_size, shuffle_buffer, number_pred,number_shift,input_dimension):
        self.window_size = window_size
        self.batch_size = batch_size
        self.shuffle_buffer = shuffle_buffer
        self.number_pred = number_pred
        self.number_shift = number_shift
        self.split_index = -number_pred - window_size
        self.input_dimension = input_dimension

    def train_valid_split(self,data):
        return data[:self.split_index], data[self.split_index:]

    def ts_data_generator(self, data, y_col=0):
        ts_data = tf.data.Dataset.from_tensor_slices(data)
        ts_data = ts_data.window(self.window_size + self.number_pred, shift=self.number_shift, drop_remainder=True)
        ts_data = ts_data.flat_map(lambda window: window.batch(self.window_size + self.number_pred))
        if self.input_dimension == 3:
            ts_data = ts_data.shuffle(self.shuffle_buffer).map(
                lambda window: (tf.reshape(window[:-self.number_pred], [self.window_size, data.shape[1]]),
                                tf.reshape(window[-self.number_pred:, y_col], [self.number_pred, 1])))
        else:
            ts_data = ts_data.shuffle(self.shuffle_buffer).map(
                lambda window: (tf.reshape(window[:-self.number_pred], [self.window_size, data.shape[1], 1]),
                                tf.reshape(window[-self.number_pred:, y_col], [self.number_pred, 1])))
        ts_data = ts_data.batch(self.batch_size).prefetch(1)
        return ts_data

    def get_tensor_dataset(self,data,y_col=0):
        train,valid = self.train_valid_split(data)
        tensor_train = self.ts_data_generator(train, y_col)
        tensor_valid = self.ts_data_generator(valid, y_col)
        return tensor_train,tensor_valid
