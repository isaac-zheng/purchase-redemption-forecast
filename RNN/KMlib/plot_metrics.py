from matplotlib import pyplot as plt


def plot_test(purchase_pred, purchase_true, redeem_pred, redeem_true):
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 2, 1)
    plt.plot(purchase_pred, label='purchase_pred')
    plt.plot(purchase_true, label='purchase_true')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(redeem_pred, label='redeem_pred')
    plt.plot(redeem_true, label='redeem_true')
    plt.legend()
    plt.show()


def plot_loss(p_history, r_history):
    plt.plot(p_history.history['loss'], label='First Train')
    plt.plot(p_history.history['val_loss'], label='First Valid')
    plt.plot(r_history.history['loss'], label='Second Train')
    plt.plot(r_history.history['val_loss'], label='Second Valid')
    plt.legend()
    plt.show()