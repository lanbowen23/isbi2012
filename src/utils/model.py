from keras.callbacks import Callback
from keras import backend as K
from math import ceil
from time import ctime
import logging
import tensorflow as tf


def jaccard_coef(y_true, y_pred):
    # print("jaccard_coef y shape: ", K.int_shape(y_pred))
    # 尝试去掉axis -2 看结果是否仍然一样： 可以的
    intersection = K.sum(y_true * y_pred, axis=[0, -1])
    # print("jaccard_coef intersection shape: ", K.int_shape(intersection))
    sum_ = K.sum(y_true + y_pred, axis=[0, -1])
    jac = (intersection + K.epsilon()) / (sum_ - intersection + K.epsilon())
    return K.mean(jac)

def jaccard_coef_int(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    intersection = K.sum(y_true * y_pred_pos, axis=[0, -1])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1])
    jac = (intersection + K.epsilon()) / (sum_ - intersection + K.epsilon())
    return K.mean(jac)

# not using this
def weighted_bce_loss(yt, yp):
    a = yt * K.log(yp + K.epsilon())
    b = (1 - yt) * K.log(1 + K.epsilon() - yp)
    m = 5
    # give m to 0(boundary label)
    # [0,1] -> [-1,0] -> [1,0] -> [m-1,0] -> [m,1]
    w = ((yt - 1) * -1) * (m - 1) + 1
    return -1 * K.mean(w * (a + b))

# https://stackoverflow.com/questions/46009619/keras-weighted-binary-crossentropy
def wce_loss(y_true, y_pred):
    zero_weight, one_weight = 0.8, 0.2
    b_ce = K.binary_crossentropy(y_true, y_pred)
    weight_vector = y_true * one_weight + (1. - y_true) * zero_weight
    weighted_b_ce = weight_vector * b_ce
    return K.mean(weighted_b_ce)

def focal_loss(y_true, y_pred):
    pt_1 = tf.where(tf.equal(y_true, 1.), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0.), y_pred, tf.zeros_like(y_pred))
    gamma = 4.
    alpha = 0.2
    pt_1 = K.clip(pt_1, 1e-3, .999)
    pt_0 = K.clip(pt_0, 1e-3, .999) # 
    return -K.sum(alpha*K.pow(1.-pt_1, gamma)*K.log(pt_1)) \
           -K.sum((1-alpha)*K.pow(pt_0, gamma)*K.log(1.-pt_0))
# https://github.com/aldi-dimara/keras-focal-loss/blob/master/focal_loss.py
def binary_focal_loss(gamma=2.0, alpha=0.2):
    """
    Formula:
        loss = -alpha_t*((1-p_t)^gamma)*log(p_t)
        
        p_t = y_pred, if y_true = 1
        p_t = 1-y_pred, otherwise
        
        alpha_t = alpha, if y_true=1
        alpha_t = 1-alpha, otherwise
        
        cross_entropy = -log(p_t)
    Parameters:
        alpha -- the same as wighting factor in balanced cross entropy
        gamma -- focusing parameter for modulating factor (1-p)
    Default value:
        gamma -- 2.0 as mentioned in the paper
        alpha -- 0.25 as mentioned in the paper
    """
    def focal_loss(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0-epsilon)

        p_t = tf.where(K.equal(y_true, 1), y_pred, 1-y_pred)
        alpha_factor = K.ones_like(y_true)*alpha
        alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1-alpha_factor)
        cross_entropy = -K.log(p_t)
        weight = alpha_t * K.pow((1-p_t), gamma)

        loss = weight * cross_entropy
        # print(K.int_shape(loss))
        # Sum the losses in mini_batch
        # loss = K.sum(loss, axis=1) 
        # print(K.int_shape(loss))
        loss = K.mean(loss)
        return loss
    return focal_loss   

def wce_plus_dice_loss(yt, yp):
    return wce_loss(yt, yp) + dice_loss_both(yt, yp)
def wce_plus_tversky_loss(yt, yp):
    return wce_loss(yt, yp) + focal_tversky_loss(yt, yp)
def focal_plus_dice_loss(yt, yp):
    return binary_focal_loss(gamma=.5)(yt, yp) + dice_loss_both(yt, yp)
    # .016*binary_focal_loss(gamma=0.2)(yt, yp)  - K.log(dice_coef_zero(yt, yp))
def focal_plus_tversky_loss(yt, yp):
    return binary_focal_loss(gamma=.5)(yt, yp) + focal_tversky_loss(yt, yp)

def dice_coef(y_true, y_pred, smooth=1.0):
    intersection = K.sum(y_true*y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)
# only this works
def dice_loss_zero(y_true, y_pred):
    return dice_coef_loss(1-y_true, 1-y_pred)
def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)
def dice_loss_both(y_true, y_pred):
    pt_1 = dice_coef(y_true, y_pred)
    pt_0 = dice_coef(1-y_true, 1-y_pred)
    return K.pow((1-pt_0), 1) + K.pow((1-pt_1), 1)

def tversky(y_true, y_pred, smooth=1.0):
    true_pos = K.sum(y_true * y_pred, axis=[1, 2, 3])
    false_neg = K.sum(y_true * (1-y_pred), axis=[1, 2, 3])
    false_pos = K.sum((1-y_true) * y_pred, axis=[1, 2, 3])
    alpha = 0.7
    return K.mean((true_pos + smooth)/(true_pos + alpha*false_neg + 
                    (1-alpha)*false_pos + smooth), axis=0)
def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)
def focal_tversky_loss(y_true,y_pred):
    pt_1 = tversky(y_true, y_pred)
    pt_0 = tversky(1-y_true, 1-y_pred)
    return K.pow((1-pt_0), 1) + K.pow((1-pt_1), 1)




def mean_diff(y_true, y_pred):
    return K.mean(y_pred) - K.mean(y_true)


def act_mean(y_true, y_pred):
    return K.mean(y_pred)


def act_min(y_true, y_pred):
    return K.min(y_pred)


def act_max(y_true, y_pred):
    return K.max(y_pred)


def act_std(y_true, y_pred):
    return K.std(y_pred)


def tru_pos(y_true, y_pred):
    axis = K.ndim(y_true) - 1
    ytam = K.argmax(y_true, axis=axis)
    ypam = K.argmax(y_pred, axis=axis)
    prod = ytam * ypam
    return K.sum(prod)


def fls_pos(y_true, y_pred):
    axis = K.ndim(y_true) - 1
    ytam = K.argmax(y_true, axis=axis)
    ypam = K.argmax(y_pred, axis=axis)
    diff = ypam - ytam
    return K.sum(K.clip(diff, 0, 1))


def tru_neg(y_true, y_pred):
    axis = K.ndim(y_true) - 1
    ytam = K.argmin(y_true, axis=axis)
    ypam = K.argmin(y_pred, axis=axis)
    prod = ytam * ypam
    return K.sum(prod)


def fls_neg(y_true, y_pred):
    axis = K.ndim(y_true) - 1
    ytam = K.argmin(y_true, axis=axis)
    ypam = K.argmin(y_pred, axis=axis)
    diff = ypam - ytam
    return K.sum(K.clip(diff, 0, 1))


def precision_onehot(y_true, y_pred):
    '''Custom implementation of the keras precision metric to work with
    one-hot encoded outputs.'''
    axis = K.ndim(y_true) - 1
    yt_flat = K.cast(K.argmax(y_true, axis=axis), 'float32')
    yp_flat = K.cast(K.argmax(y_pred, axis=axis), 'float32')
    true_positives = K.sum(K.round(K.clip(yt_flat * yp_flat, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(yp_flat, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall_onehot(y_true, y_pred):
    '''Custom implementation of the keras recall metric to work with
        one-hot encoded outputs.'''
    axis = K.ndim(y_true) - 1
    yt_flat = K.cast(K.argmax(y_true, axis=axis), 'float32')
    yp_flat = K.cast(K.argmax(y_pred, axis=axis), 'float32')
    true_positives = K.sum(K.round(K.clip(yt_flat * yp_flat, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(yt_flat, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def fmeasure_onehot(y_true, y_pred):
    '''Custom implementation of the keras fmeasure metric to work with
    one-hot encoded outputs.'''
    p = precision_onehot(y_true, y_pred)
    r = recall_onehot(y_true, y_pred)
    return 2 * (p * r) / (p + r + K.epsilon())


class KerasHistoryPlotCallback(Callback):

    def on_train_begin(self, logs={}):
        self.logs = {}

    def on_epoch_end(self, epoch, logs={}):

        if hasattr(self, 'file_name'):
            import matplotlib
            matplotlib.use('agg')

        import matplotlib.pyplot as plt

        if len(self.logs) == 0:
            self.logs = {key: [] for key in logs.keys()}

        for key, val in logs.items():
            self.logs[key].append(val)

        nb_metrics = len([k for k in self.logs.keys()
                          if not k.startswith('val')])
        nb_col = 6
        nb_row = int(ceil(nb_metrics * 1.0 / nb_col))
        fig, axs = plt.subplots(nb_row, nb_col, figsize=(
            min(nb_col * 3, 12), 3 * nb_row))
        for idx, ax in enumerate(fig.axes):
            if idx >= len(self.logs):
                ax.axis('off')
                continue
            key = sorted(self.logs.keys())[idx]
            if key.startswith('val_'):
                continue
            ax.set_title(key)
            ax.plot(self.logs[key], label='TR')
            val_key = 'val_%s' % key
            if val_key in self.logs:
                ax.plot(self.logs[val_key], label='VL')
            ax.legend()

        plt.suptitle('Epoch %d: %s' % (epoch, ctime()), y=1.10)
        plt.tight_layout(pad=0.8, w_pad=0.8, h_pad=1.0)

        if hasattr(self, 'file_name'):
            plt.savefig(self.file_name)
        else:
            plt.show()


class KerasSimpleLoggerCallback(Callback):

    def on_train_begin(self, logs={}):
        self.prev_logs = None
        return

    def on_epoch_end(self, epoch, logs={}):

        logger = logging.getLogger(__name__)

        if self.prev_logs == None:
            for key, val in logs.items():
                logger.info('%15s: %.5lf' % (key, val))
        else:
            for key, val in logs.items():
                diff = val - self.prev_logs[key]
                logger.info('%20s: %15.4lf %5s %15.4lf' %
                            (key, val, '+' if diff > 0 else '-', abs(diff)))

        self.prev_logs = logs
