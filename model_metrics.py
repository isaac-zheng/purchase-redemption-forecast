import math
import numpy as np


def daily_score(error):
    """
    :param error: daily error of purchase or redeem
    :return: daily score
    """
    return 10 * math.e ** -(10 * (error))


def get_score(y_true, y_pred):
    """
    :param y_true: array with size = 30. 30 day ground truth
    :param y_pred: array with size = 30. 30 day prediction
    :return: 30 day total score for purchase or redemption
    """
    y_true, y_pred = y_true.reshape(-1, ), y_pred.reshape(-1)
    error = np.abs(y_true - y_pred) / y_true
    return sum(list(map(daily_score, error)))


def total_score(purchase_true, purchase_pred, redeem_true, redeem_pred):
    """
    :param purchase_true: array with size = 30. y_true for purchase
    :param purchase_pred: array with size = 30. y_pred for purchase
    :param redeem_true: array with size = 30. y_true for redeem
    :param redeem_pred: array with size = 30. y_pred for redeem
    :return: Final score of purchase and redemption.
    """
    pscore = get_score(purchase_true, purchase_pred)
    rscore = get_score(redeem_true, redeem_pred)
    return 0.45 * pscore + 0.55 * rscore


if __name__ == '__main__':
    purchase_true = np.arange(10, 40)
    purchase_pred = np.arange(11, 41)
    redeem_true = np.arange(20, 50)
    redeem_pred = np.arange(21, 51)

    print(total_score(purchase_true, purchase_pred, redeem_true, redeem_pred))
