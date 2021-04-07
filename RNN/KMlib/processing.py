import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import random
from matplotlib import pyplot as plt


def get_test_data(df, p_scaler, rc_scaler, rt_scaler):
    p_data = np.array(df['total_purchase_amt'])
    rc_data = np.array(df['consume_amt'])
    rt_data = np.array(df['transfer_amt'])
    p_test = p_scaler.transform(p_data.reshape(p_data.shape[0], -1))
    rc_test = rc_scaler.transform(rc_data.reshape(rc_data.shape[0], -1))
    rt_test = rt_scaler.transform(rt_data.reshape(rc_data.shape[0], -1))
    return p_test, rc_test, rt_test


def pr_scaling(df_poor):
    p_data = np.array(df_poor['total_purchase_amt'])
    rc_data = np.array(df_poor['consume_amt'])
    rt_data = np.array(df_poor['transfer_amt'])
    p_scaler = MinMaxScaler(feature_range=(0, 1))
    rc_scaler = MinMaxScaler(feature_range=(0, 1))
    rt_scaler = MinMaxScaler(feature_range=(0, 1))
    p_data = p_scaler.fit_transform(p_data.reshape(-1, 1)).reshape(len(p_data), 1)
    rc_data = rc_scaler.fit_transform(rc_data.reshape(-1, 1)).reshape(len(rc_data), 1)
    rt_data = rt_scaler.fit_transform(rt_data.reshape(-1, 1)).reshape(len(rt_data), 1)
    return p_data, rc_data, rt_data, p_scaler, rc_scaler, rt_scaler


def x_scaling(df):
    data = df.values
    x_scaler = MinMaxScaler(feature_range=(0, 1))
    x_data = x_scaler.fit_transform(data)
    return x_data, x_scaler


class Processer(object):
    def __init__(self, window_size, batch_size, shuffle_buffer, number_pred, number_shift, input_dimension):
        """
        :param window_size: Size of input X
        :param batch_size: batch size for training
        :param number_pred: number of predictions y
        :param number_shift: shift amount when generating window dataset
        :param input_dimension: the required dimensions of model input
        """
        self.window_size = window_size
        self.batch_size = batch_size
        self.shuffle_buffer = shuffle_buffer
        self.number_pred = number_pred
        self.number_shift = number_shift
        self.split_index = -number_pred - window_size
        self.input_dimension = input_dimension

    def train_valid_split(self, data):
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

    def get_tensor_dataset(self, data, y_col=0):
        train, valid = self.train_valid_split(data)
        tensor_train = self.ts_data_generator(train, y_col)
        tensor_valid = self.ts_data_generator(valid, y_col)
        return tensor_train, tensor_valid


class WindowGenerator():
    """
    Modified Class from TF guide, tf version 2.4.1 required
    """
    def __init__(self, input_width, label_width, shift, batch_size,train_df, val_df, test_df, label_columns=None):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.batch_size = batch_size

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def plot(self, model=None, plot_col='total_purchase_amt', max_subplots=3):
        inputs, labels = self.plot_example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n + 1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                     label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                            marker='X', edgecolors='k', label='Predictions',
                            c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Time [h]')

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=self.batch_size, )

        ds = ds.map(self.split_window)

        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        result = getattr(self, '_example', None)
        if result is None:
            result = next(iter(self.train))
            self._example = result
        return result

if random.random() > 0.999:
    print(r"""
    ////////////////////////////////////////////////////////////////////
    //                          _ooOoo_                               //
    //                         o8888888o                              //
    //                         88" . "88                              //
    //                         (| ^_^ |)                              //
    //                         O\  =  /O                              //
    //                      ____/`---'\____                           //
    //                    .'  \\|     |//  `.                         //
    //                   /  \\|||  :  |||//  \                        //
    //                  /  _||||| -:- |||||-  \                       //
    //                  |   | \\\  -  /// |   |                       //
    //                  | \_|  ''\---/''  |   |                       //
    //                  \  .-\__  `-`  ___/-. /                       //
    //                ___`. .'  /--.--\  `. . ___                     //
    //              ."" '<  `.___\_<|>_/___.'  >'"".                  //
    //            | | :  `- \`.;`\ _ /`;.`/ - ` : | |                 //
    //            \  \ `-.   \_ __\ /__ _/   .-` /  /                 //
    //      ========`-.____`-.___\_____/___.-`____.-'========         //
    //                           `=---='                              //
    //      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^        //
    //             佛祖保佑       永无BUG     光速炼丹                 //
    ////////////////////////////////////////////////////////////////////
    """)
