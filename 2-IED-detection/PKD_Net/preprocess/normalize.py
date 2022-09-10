#!/usr/bin/env python2.7
#-*- coding:utf-8 -*-
# @Time    : 4/17/19 11:55 AM
# @Author  : CMR
# @File    : normalize.py
# @Software: PyCharm
# @Script to:
#   -
import numpy as np
from copy import deepcopy

def standardscaler(data,range=[-1,1]):
    """
    将MEG数据归一化到指定范围（仅用于MEG数据）;对每个通道分别计算min,max,用于scale.
    :param data: n_Batch x n_channel x n_timestamp
    :param range:
    :return:
    """
    data = deepcopy(data)
    if range[0] > range[1]:
        raise ValueError("the range is not right...")
    grads = data[:,0:26,:]
    # print("grad shape:",grads.shape)
    mags = data[:,26:39,:]
    # print("mags shape:", mags.shape)
    grads_min_value = np.min(grads,axis=-1)
    grads_max_value = np.max(grads,axis=-1)

    grads_min_value = np.expand_dims(grads_min_value,axis=-1)
    grads_min_value = np.repeat(grads_min_value,data.shape[-1],axis=-1)
    grads_max_value = np.expand_dims(grads_max_value,axis=-1)
    grads_max_value = np.repeat(grads_max_value,data.shape[-1],axis=-1)

    # print("grads min",grads_min_value.shape)
    # print("grads max",grads_max_value)
    if np.any(np.where(grads_max_value==grads_min_value)):
        print("some of the grads max value equal to grads min value...")
        grads_max_value[np.where(grads_max_value==grads_min_value)] += 1e-20

    mags_min_value = np.min(mags,axis=-1)
    mags_max_value = np.max(mags,axis=-1)

    mags_min_value = np.expand_dims(mags_min_value,axis=-1)
    mags_min_value = np.repeat(mags_min_value,data.shape[-1],axis=-1)
    mags_max_value = np.expand_dims(mags_max_value,axis=-1)
    mags_max_value = np.repeat(mags_max_value,data.shape[-1],axis=-1)

    # print("mags min",mags_min_value.shape)
    # print("mags max",mags_max_value)
    if np.any(np.where(mags_max_value==mags_min_value)):
        print("some of the mags max value equal to mags min value...")
        mags_max_value[np.where(mags_max_value==mags_min_value)] += 1e-20

    data[:,0:26,:] =(range[1]-range[0])*(data[:,0:26,:] - grads_min_value)/(grads_max_value-grads_min_value)+range[0]
    data[:,26:39,:] =(range[1]-range[0])*(data[:,26:39,:] - mags_min_value)/(mags_max_value-mags_min_value)+range[0]

    return data


def gaussscaler(data):
    """
    将MEG数据归一化到高斯（仅用于MEG数据）;对每个通道分别计算min,max,用于scale.
    :param data: n_Batch x n_channel x n_timestamp
    :param range:
    :return:
    """
    meg_mean = data.mean(axis=1)
    meg_mean = np.expand_dims(meg_mean, axis=1)
    meg_mean = np.repeat(meg_mean, data.shape[1], axis=1)
    meg_std = data.std(axis=1)
    meg_std = np.expand_dims(meg_std, axis=1)
    meg_std = np.repeat(meg_std, data.shape[1], axis=1)

    data = (data - meg_mean) / (meg_std + 1e-7)

    return data

if __name__=='__main__':
    datas = np.random.random((3,39,6))
    # datas[0,0:26,2] = -2
    # datas[0,0:26,3] = 2
    # datas[0, 26:39, 4] = 2
    # datas[1,0:26,2] = 2
    # datas[1,26:39,4] = 3
    # datas[2,0:26,2] = 2
    # datas[2,26:39,4] = 3
    print(datas)
    standard_data = guassscaler(datas)
