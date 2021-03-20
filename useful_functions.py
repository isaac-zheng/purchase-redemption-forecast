import math
import numpy as np
import datetime
import pandas as pd
def split_data_aug(data):
    train = data[(datetime.datetime(2014,4,1) <= data['date']) & (data['date'] < datetime.datetime(2014,8,1))]
    test = data[(datetime.datetime(2014,8,1) <= data['date']) & (data['date'] < datetime.datetime(2014,9,1))]
    return train, test

def split_data_sep(data):
    train = data[(datetime.datetime(2014,4,1) <= data['date']) & (data['date'] < datetime.datetime(2014,9,1))]
    test = data[(datetime.datetime(2014,9,1) <= data['date']) & (data['date'] < datetime.datetime(2014,10,1))]
    return train, test

def error(y_true, y_pred):
    return np.abs(y_true - y_pred) / np.abs(y_true)


def total_score(purchase_pred, purchase_true, redeem_pred, redeem_true, threshold=0.3):
    return sum(map(lambda x : 0 if x > 0.3 else np.exp(-x/threshold)*10, error(purchase_true, purchase_pred))) * 0.45 + sum(map(lambda x : 0 if x > 0.3 else np.exp(-x/threshold)*10, error(redeem_true, redeem_pred))) * 0.55


def get_submit_data(purchase_pred,redeem_pred):
    df = pd.DataFrame([[x for x in range(20140901,20140931)],purchase_pred,redeem_pred]).T
    df[0]= df[0].astype('int')
    df.loc[:,1:2] = df.loc[:,1:2].apply(lambda x: round(x,2))
    df.to_csv('dataset/tc_comp_predict_table.csv',index = False,header = None)
#     df.to_csv(f'history/{datetime.time}.csv')
    return df

if __name__ == '__main__':
    purchase_true = np.arange(10, 40)
    purchase_pred = np.arange(11, 41)
    redeem_true = np.arange(20, 50)
    redeem_pred = np.arange(21, 51)

    print(total_score(purchase_true, purchase_pred, redeem_true, redeem_pred))
