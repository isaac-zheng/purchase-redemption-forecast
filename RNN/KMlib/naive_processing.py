import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler


def train_scaler(raw_x_train, raw_y_train):
    x_scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaler = MinMaxScaler(feature_range=(0, 1))
    x_scaler.fit(raw_x_train)
    y_scaler.fit(raw_y_train)
    return x_scaler, y_scaler


def get_window_data(df, window_size, number_split):
    df_train = df.loc[:"2014-08-01"]
    df_x_test = df.loc[:"2014-08-01"].iloc[-window_size:]
    dataset = df_train.values
    tscv = TimeSeriesSplit(number_split, max_train_size=window_size, test_size=30)
    X, Y = [], []
    i = 0
    raw_x_train = df_train
    raw_y_train = df_train.values[:, :2]
    x_scaler, y_scaler = train_scaler(raw_x_train, raw_y_train)
    for x_index, y_index in tscv.split(dataset):
        a = df_train.iloc[x_index].index
        b = df_train.iloc[y_index].index
        print(f"X[{i}] from: {a[0]} to {a[-1]}", end="\t")
        X.append(x_scaler.transform(dataset[x_index]))
        print(f"Y[{i}] from: {b[0]} to {b[-1]}")
        Y.append(y_scaler.transform(dataset[y_index][:, :2]))
        i += 1
    x_train, y_train = np.array(X), np.array(Y)
    x_valid = x_scaler.transform(df_x_test)
    y_valid = y_scaler.transform(df.loc["2014-08-02":"2014-08-31"].iloc[:, :2])
    return x_train, x_valid, y_train, y_valid, x_scaler, y_scaler


def predict(tp, tr, p_valid, t_valid, y_scaler):
    p = tp.predict(p_valid)
    r = tr.predict(t_valid)
    all_pred = np.concatenate([p, r]).T
    y_test = y_scaler.inverse_transform(all_pred)
    purchase_pred = y_test[:, 0]
    redeem_pred = y_test[:, 1]
    return purchase_pred, redeem_pred


def get_submit_data(purchase_pred, redeem_pred, name=0):
    if not name:
        name = str(datetime.now().date())
    df = pd.DataFrame([[x for x in range(20140901, 20140931)], purchase_pred, redeem_pred]).T
    df[0] = df[0].astype('int')
    df.loc[:, 1:2] = df.loc[:, 1:2].apply(lambda x: round(x, 2))
    df.to_csv('tc_comp_predict_table.csv', index=False, header=None)
    #     df.to_csv(f'history/{name}.csv',index = False,header = None)
    return df
