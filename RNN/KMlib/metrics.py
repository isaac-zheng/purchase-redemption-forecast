import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import TimeSeriesSplit
from datetime import datetime
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def purchase_error(y_true, y_pred):
    return np.abs(y_true - y_pred) / np.abs(y_true)


def redeem_error(y_true, y_pred):
    return (y_true - y_pred) / np.abs(y_true)


def total_score(purchase_pred, purchase_true, redeem_pred, redeem_true, threshold=0.15):
    def transfer(x):
        if abs(x) > 0.3:
            return 0
        if x < 0:  # overpredict
            return np.exp(x / threshold) * 10
        else:
            return np.exp(1.5 * -x / threshold) * 10

    redeem_score = sum(map(transfer, redeem_error(redeem_true, redeem_pred))) * 0.55
    return sum(map(lambda x: 0 if x > 0.3 else np.exp(-x / threshold) * 10,
                   purchase_error(purchase_true, purchase_pred))) * 0.45 + redeem_score


