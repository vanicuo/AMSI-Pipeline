# -*- coding:utf-8 -*-

import numpy as np
from scipy import stats
import pickle
# 保存处理后的数据集
# spike train data 与 non_spike train data
# peek data 使用不同的sigma值计算label，并保存
# 全部保存好，在训练的时候根据list的索引选择训练集验证集即可


sub_names = ['sub01', 'sub02', 'sub03', 'sub04', 'sub05',
             'sub06', 'sub07', 'sub08', 'sub09', 'sub10',
             'sub11', 'sub12', 'sub13', 'sub14', 'sub15',
             'sub16', 'sub17', 'sub18', 'sub19', 'sub20',
             'sub21', 'sub22', 'sub23', 'sub24', 'sub25',
             'sub26', 'sub27', 'sub28', 'sub29', 'sub30',
             'sub31', 'sub32', 'sub33', 'sub34', 'sub35']

def convert_peek_to_norm_categorical_py2(label_index, segment_length=300, sigma=0.01, **kwargs):
    assert label_index >= 0, 'label index exceed.'
    assert label_index < segment_length, 'label index exceed.'
    #  正态分布的均值,用于调整生成分布峰值所在位置
    mu = (label_index - segment_length / 2.) / (segment_length / 2.)
    x = np.linspace(-1, 1, segment_length)
    y = stats.norm(mu, sigma).pdf(x)
    std_y = y / np.max(y)
    std_y = np.expand_dims(std_y, axis=-1)
    std_y = std_y.T
    return std_y

def load_dataset(index):
    """
    根据sub_names的index来获取数据集
    :param index:
    :return:
    """
    trains = list()
    peeks = list()
    labels = list()
    # temp=np.empty((0,))
    for i in index:
        print(str(sub_names[i]))
        # sub_basename = './preprocess/'+str(sub_names[i])
        # sub_basename = '../preprocess/data_Spike_DNet/' + str(sub_names[i])
        sub_basename = '/data1/zhengli/meg_classification/preprocess/data_Spike_DNet/' + str(sub_names[i])
        spike = np.load(sub_basename + '_meg_spike_sets.npy')
        peek = np.load(sub_basename + '_meg_segment_peeks_sets.npy')
        # print("spike shape:", np.size(spike, axis=0))
        train = np.expand_dims(spike, axis=3)
        # print("train shape:", train.shape)

        peeks_tmp = np.empty((0, 300))
        for j in peek:
            # print("----------i result is:",i)
            peek_categ = convert_peek_to_norm_categorical_py2(j, segment_length=300, sigma=0.01,
                                                              labels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1, 1, 1])
            # print("----------before convert vs. after convert is:", peek_categ.shape,i, np.argmax(peek_categ, axis=1))
            # print("----------after convert is:", np.argmax(peek_categ, axis=0).shape)
            peeks_tmp = np.concatenate((peeks_tmp, peek_categ), axis=0)
            # break
        # peek = np.concatenate((peek, non_peek), axis=0)
        # print("train shape",train.shape)
        # print("label shape",label.shape)
        trains.append(train)
        peeks.append(peeks_tmp)
    # print("loading_data shape:", trains.shape)
    # print("loading_labels shape:", np.array(labels).shape)
    # print("segment peeks shape:", peeks.shape)
    return trains, labels, peeks


def load_non_dataset(index):
    """
    根据sub_names的index来获取数据集
    :param index:
    :return:
    """
    trains = list()
    peeks = list()
    # temp=np.empty((0,))
    for i in index:
        print(str(sub_names[i]))
        # sub_basename = '../preprocess/data_Spike_DNet/' + str(sub_names[i])
        sub_basename = '/data1/zhengli/meg_classification/preprocess/data_Spike_DNet/' + str(sub_names[i])
        spike = np.load(sub_basename + '_meg_non_spike_sets.npy')
        non_spike_label = np.zeros((spike.shape[0], 300))
        train = np.expand_dims(spike, axis=3)
        trains.append(train)
        peeks.append(non_spike_label)
    # print("loading_data shape:", trains.shape)
    # print("loading_labels shape:", np.array(labels).shape)
    # print("segment peeks shape:", peeks.shape)
    return trains, peeks


def load_dataset_peek(index, sigma=0.02):
    peeks = list()
    for i in index:
        print(str(sub_names[i]))
        # sub_basename = '../preprocess/data_Spike_DNet/' + str(sub_names[i])
        sub_basename = '/data1/zhengli/meg_classification/preprocess/data_Spike_DNet/' + str(sub_names[i])
        peek = np.load(sub_basename + '_meg_segment_peeks_sets.npy')
        peeks_tmp = np.empty((0, 300))
        for j in peek:
            peek_categ = convert_peek_to_norm_categorical_py2(j, segment_length=300, sigma=sigma)
            peeks_tmp = np.concatenate((peeks_tmp, peek_categ), axis=0)
        peeks.append(peeks_tmp)
    return peeks


train_index = range(0, 35)
train_index = range(0,3)
# 下面是保存train data的代码，运行一次即可
meg_train_ori, meg_train_label, train_peeks = load_dataset(train_index)

with open('../preprocess/meg_train_ori.pkl', 'wb') as f:
    pickle.dump(meg_train_ori, f)

# 似乎meg_train_label并没有用
with open('../preprocess/meg_train_label.pkl', 'wb') as f:
    pickle.dump(meg_train_label, f)

with open('../preprocess/train_peeks.pkl', 'wb') as f:
    pickle.dump(train_peeks, f)

# 下面是保存non_spike train data的代码，运行一次即可
# meg_non_train_ori, train_non_peeks = load_non_dataset(train_index)
# with open('../preprocess/meg_non_train_ori.pkl', 'wb') as f:
#     pickle.dump(meg_non_train_ori, f)
# with open('../preprocess/train_non_peeks.pkl', 'wb') as f:
#     pickle.dump(train_non_peeks, f)

# 下面是保存train_peeks的代码，调整sigma值会得到不同的结果，并且保存成不同的文件
# 文件最后为sgmXXX, XXX为sigma取值，010代表的是0.10
# sigma = 0.1
# train_peeks = load_dataset_peek(train_index, sigma=sigma)
# with open('../preprocess/train_peeks_sgm%03d.pkl' % (sigma*100), 'wb') as f:
#     pickle.dump(train_peeks, f)

