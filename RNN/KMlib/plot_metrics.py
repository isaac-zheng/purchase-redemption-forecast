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
    try:
        plt.figure(figsize=(12, 5))
        plt.plot(p_history.history['loss'][5:], label='First Train')
        plt.plot(p_history.history['val_loss'][5:], label='First Valid')
        plt.plot(r_history.history['loss'][5:], label='Second Train')
        plt.plot(r_history.history['val_loss'][5:], label='Second Valid')
        plt.legend()
        plt.title('loss after 5 epochs')
        plt.show()
    except:
        print('Please increase Epochs')