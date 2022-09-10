# coding=utf-8
# 2020.11.02
# CMR
# 包括了loss函数和计算指标

# dice loss可以尝试使用，当时的label的正态分布函数的sigma值比较低，因此效果不好
# 在提高了sigma值之后没有测试dice loss的效果，值得测试以下

# 关于peak bias的计算指标并不准确，因为并不是把整个数据集放在一起计算的
# 而是每个mini batch各自计算，再取平均值
# 因此需要使用Callback，在调用on_epoch_end的函数，将验证集直接输入进行计算
#

from keras import backend as K
import tensorflow as tf
import keras
from keras.callbacks import Callback
import numpy as np

def dice():
    def dice_fn(y_true, y_pred, smooth=0.00001):
        # y_true = y_true / (K.max(y_true) - K.min(y_true))
        # y_pred = y_pred / (K.max(y_pred) - K.min(y_pred))
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    dice_fn.__name__ = 'dice'
    return dice_fn

def dice_loss(y_true, y_pred, smooth=0.00001):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1- ((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))

def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def peak_bias_max(y_true, y_pred):
    return K.max(K.abs(K.argmax(y_true,axis=1) - K.argmax(y_pred, axis=1)))

def peak_bias_min(y_true, y_pred):
    return K.min(K.abs(K.argmax(y_true,axis=1) - K.argmax(y_pred, axis=1)))

def peak_bias_mean(y_true, y_pred):
    return K.mean(K.abs(K.argmax(y_true,axis=1) - K.argmax(y_pred, axis=1)))

def peak_bias_std(y_true, y_pred):
    return K.std(tf.cast(K.abs(K.argmax(y_true,axis=1) - K.argmax(y_pred, axis=1)), dtype=tf.float32))

def peak_bias(y_true, y_pred):
    return K.argmax(y_true,axis=1) - K.argmax(y_pred, axis=1)

def peak_bias_metrcis(y_true, y_pred):
    bias_arr = np.argmax(y_true, axis=1) - np.argmax(y_pred, axis=1)
    bias_abs = np.abs(bias_arr)
    bias_max = np.max(bias_abs)
    bias_min = np.min(bias_abs)
    bias_mean = np.mean(bias_abs)
    bias_std = np.std(bias_abs)
    return bias_arr, bias_max, bias_min, bias_mean, bias_std


class peak_callback(keras.callbacks.Callback):
    def __init__(self, validation_data):
        super(peak_callback, self).__init__()
        self.metrcis_dict = dict()
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs=None):
        #
        # predict_proba = np.asarray(self.model.predict(self.validation_data[0]))
        # print("\n validation shape:", self.validation_data[0].shape)
        # target = self.validation_data[1]
        # predict = np.round(predict_proba)
        y_pred = self.model.predict(self.validation_data[0], verbose=0)
        bias_arr, bias_max, bias_min, bias_mean, bias_std = peak_bias_metrcis(y_pred, self.validation_data[1])
        self.metrcis_dict['bias_arr'] = bias_arr
        self.metrcis_dict['bias_max'] = bias_max
        self.metrcis_dict['bias_min'] = bias_min
        self.metrcis_dict['bias_mean'] = bias_mean
        self.metrcis_dict['bias_std'] = bias_std

