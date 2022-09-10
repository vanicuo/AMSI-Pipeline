# -*- coding:utf-8 -*-
# @Time    : 10/28/20 7:21 PM
# @Author  : CMR
# @File    : main_Peak_detect_cv_junxian.py
# @Software: PyCharm
# @Script to:

# 以下为实验记录，保存在/logs/Peak_dnet_junxian中，第n次代表按照时间顺序排列的第n个文件夹
# 第01次为unet_1D_v2，same padding的unet, sigma=0.01
# 第02次为unet_1D_v1，valid padding的unet, sigma=0.01
# 第03次为unet_1D_v5，不同kernel的unet, sigma=0.01  20, 10, 5, 3, 5, 10, 20
# 第04次为unet_1D_v5，不同kernel的unet, sigma=0.10  20, 10, 5, 3, 5, 10, 20
# 第05次为unet_1D_v6，加入了resblock, sigma=0.10  20, 10, 5, 3, 5, 10, 20
# 第06次为unet_1D_v5，不同kernel的unet, sigma=0.20  20, 10, 5, 3, 5, 10, 20
# 第07次为unet_1D_v5，大尺度的卷积核, sigma=0.10  40, 30, 20, 10, 20, 30, 40
# 第08次为unet_1D_v5，maxpooling, 大尺度的卷积核, sigma=0.10  40, 30, 20, 10, 20, 30, 40
# 第09次为unet_1D_v6，加入了resblock,大尺度的卷积核, sigma=0.10  40, 30, 20, 10, 20, 30, 40, 加入了nonspike数据
# 2020_11_19_02_12_43：Peak_DNet, 作为参考，sigma=0.01
# 2020_11_19_02_12_43：Peak_DNet, 作为参考，sigma=0.10
# 2020_11_19_02_12_43：Peak_DNet, 作为参考，sigma=0.01
# 第11次为unet_1D_v4, skip connection直接展开，sigma=0.10
# 第12次为unet_1D_v4, skip connection在feature map维度上进行global avg，sigma=0.10
# 第13次为unet_1D_v4, skip connection在meg channel维度上进行global avg，sigma=0.10

import time
import os
import pickle
import shutil

import numpy as np
from keras.optimizers import *
import junxian_lib.peak_metrics as junxian_metrics
import net.unet_1D as unet
from metrics import recall, precision, f1_score
# from models_for_peak import Peak_DNet_junxian, Peak_DNet
import preprocess.normalize as norm
os.environ["CUDA_VISIBLE_DEVICES"] = '0,2,3'
# 记录实验时间，为实验记录的文件夹名
time_now = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
print(time_now)
# define work path
path_output = '/data2/zhengli/peak_detection/logs/4KMEG'  # 实验记录的根文件夹
# 实验记录的临时文件夹，只有在整个代码跑完后才会将记录移动到实验记录的根文件夹，即path_output
path_output_temp = '/data2/zhengli/peak_detection/logs/4KMEG/temp'
# 定义一些文件夹名，并且创建
if not os.path.exists(path_output):
    os.makedirs(path_output)
if not os.path.exists(path_output_temp):
    os.makedirs(path_output_temp)

path_log = os.path.join(path_output_temp, time_now)
path_csv = os.path.join(path_log, 'csv')
path_models = os.path.join(path_log, 'models')
path_outputs = os.path.join(path_log, 'output')
if not os.path.exists(path_log):
    os.makedirs(path_log)
    os.makedirs(path_csv)
    os.makedirs(path_models)
    os.makedirs(path_outputs)
# Load dataset
# sub_names = ['sub01', 'sub11', 'sub12', 'sub14', 'sub15',
#              'sub16', 'sub17', 'sub20', 'sub21', 'sub22',
#              'sub25', 'sub26', 'sub27', 'sub31', 'sub32',
#              'sub33', 'sub34', 'sub35', 'sub36', 'sub37',
#              'sub02', 'sub03', 'sub04', 'sub06', 'sub09',
#              'sub10', 'sub13', 'sub18', 'sub23', 'sub24',
#              'sub09', 'sub28', 'sub29', 'sub30', 'sub05']
# num_case = len(sub_names)  # case的数量

sheet_hash = {
    '1st': [
        'MEG1002', 'MEG1080', 'MEG1097', 'MEG1121', 'MEG1126', 'MEG1179', 'MEG0847', 'MEG0850', 'MEG0860',
        'MEG0863', 'MEG0871', 'MEG0875', 'MEG0880', 'MEG0890', 'MEG0896', 'MEG0913', 'MEG0919', 'MEG0922',
        'MEG0952', 'MEG0962', 'MEG0966', 'MEG0969', 'MEG0970', 'MEG0992'
    ],
    '2nd': [
        'MEG0341', 'MEG0382', 'MEG0446', 'MEG0013', 'MEG0037', 'MEG0085', 'MEG0092', 'MEG0109', 'MEG0212',
        'MEG0213'
    ],  # , 'MEG0218'  'MEG0223'
    '3rd': [
        'MEG0512', 'MEG0550', 'MEG0567', 'MEG0571', 'MEG0576', 'MEG0579', 'MEG0584', 'MEG0587', 'MEG0604',
        'MEG0610', 'MEG0615'
    ],
    '4th': [
        'MEG0622', 'MEG0626', 'MEG0641', 'MEG0645', 'MEG0651', 'MEG0652', 'MEG0658', 'MEG0677', 'MEG0694',
        'MEG0700'
    ]
}

sheet_names = []
for batch in ['1st', '2nd', '3rd', '4th']:
    data_dir1 = f'/data1/luoshen/Files/4KMEG_39_{batch}batch/train_data/'
    sheet_names.extend([os.path.join(data_dir1, sub) for sub in sheet_hash[batch]])
    sub_names_train = np.array(sheet_names)
num_case = len(sheet_names)  # case的数量
print('Loading Dataset')
len_raw_segment = 300  # 原始segment长度
# 读取数据集，label，
with open('/data2/zhengli/peak_detection/preprocess/meg_train_ori.pkl', 'rb') as f:
    data_meg = pickle.load(f)
with open('/data2/zhengli/peak_detection/preprocess/train_peeks_sgm010.pkl', 'rb') as f:
    data_peeks = pickle.load(f)
# with open('./preprocess/meg_non_train_ori.pkl', 'rb') as f:
#     data_meg_non = pickle.load(f)
# with open('./preprocess/train_non_peeks.pkl', 'rb') as f:
#     data_peeks_non = pickle.load(f)
print('Finished !!')

# 设置超参数
# Set hyper param
# hyper param for deep learning
epochs = 20
learning_rate = 0.001
learning_rate_decay = 0.0001
batch_size = 256
# 为300时，会使用一些网络 # Peak_DNet和unet_1D
# 为296时，会使用其他网络
len_segment = 296    # 300 or 296

# 计算需要裁剪的长度
len_crop = int((len_raw_segment -len_segment) / 2)
# 训练时计算的指标，其实意义不大，最多dice值有意义，其他的意义都不大
list_metrics = ['accuracy', 'categorical_accuracy', recall, precision, f1_score, junxian_metrics.dice()]
                # junxian_metrics.peak_bias_max, junxian_metrics.peak_bias_min,
                # junxian_metrics.peak_bias_mean, junxian_metrics.peak_bias_std,
                # junxian_metrics.peak_bias]
# 没有使用focal loss
# # focal loss
# def focal_loss(y_true, y_pred):
#     gamma = 2.
#     alpha = .25
#     pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
#     pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
#     return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum(
#         (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

# 开始7折交叉验证
# CrossValidation  7-fold
num_cv = 5
num_valid = int(num_case / num_cv) # 验证集的case数量
list_out = []       # 保存输出的list
list_pred_cv = []   # 保存网络直接预测的list
for icv in range(num_cv):
    print('Cross Validation %02d' % (icv + 1))
    train_index = range(num_case)                               # 训练集
    train_index = list(train_index)
    valid_index = range(icv * num_valid, (icv + 1) * num_valid) # 验证集，第一次为(0,4)，第二次为(5,9)，...

    for cases in valid_index:
        print(cases)
        train_index.remove(cases) # 删除训练集中的验证集，得到正确的训练集

    # 设置实验使用的网络，根据之前len_segment进行判断需要两个地方设置同步
    if len_segment == 300:
        # model = Peak_DNet((39, 300, 1))
        model = unet.unet_1D((len_segment, 39), base_feature=4)
    else:
        model = unet.unet_1D_v5((len_segment, 39), base_feature=4)

    # 准备数据，根据训练集，验证集的label，设置好数据
    print('Prepare dataset !!')
    train_data = np.empty((0, 39, len_segment, 1))
    valid_data = np.empty((0, 39, len_segment, 1))
    train_peeks = np.empty((0, len_segment))
    valid_peeks = np.empty((0, len_segment))

    # 训练集数据
    for cases in train_index:
        if len_crop == 0:
            train_data = np.concatenate((train_data, data_meg[cases]), axis=0)
            train_peeks = np.concatenate((train_peeks, data_peeks[cases]), axis=0)
        else:
            train_data = np.concatenate((train_data, data_meg[cases][:, :, len_crop:-len_crop, :]), axis=0)
            train_peeks = np.concatenate((train_peeks, data_peeks[cases][:, len_crop:-len_crop]), axis=0)

    # 训练集数据加上non_spike数据，不加上则可以注释下面代码
    # for cases in range(35):
    #     if len_crop == 0:
    #         train_data = np.concatenate((train_data, data_meg_non[cases]), axis=0)
    #         train_peeks = np.concatenate((train_peeks, data_peeks_non[cases]), axis=0)
    #     else:
    #         train_data = np.concatenate((train_data, data_meg_non[cases][:, :, len_crop:-len_crop, :]), axis=0)
    #         train_peeks = np.concatenate((train_peeks, data_peeks_non[cases][:, len_crop:-len_crop]), axis=0)

    # 验证集数据
    for cases in valid_index:
        if len_crop == 0:
            valid_data = np.concatenate((valid_data, data_meg[cases]), axis=0)
            valid_peeks = np.concatenate((valid_peeks, data_peeks[cases]), axis=0)
        else:
            valid_data = np.concatenate((valid_data, data_meg[cases][:, :, len_crop:-len_crop, :]), axis=0)
            valid_peeks = np.concatenate((valid_peeks, data_peeks[cases][:, len_crop:-len_crop]), axis=0)
    # train_data.shape: (0, 39, len_segment, 1)
    # 对数据进行归一化，目前使用的是gauss归一化
    train_data = np.squeeze(train_data)                 # [?, 39, len_segment]
    train_data = norm.gaussscaler(train_data)           # [?, 39, len_segment]
    valid_data = np.squeeze(valid_data)
    valid_data = norm.gaussscaler(valid_data)

    # 以下比较关键，不同网络输入的数据shape不一致，需要transpose或者expand_dims
    # 经过归一化后的数据大小为[?, 39, len_segment]（?为batch的大小，不需要考虑），因此需要根据网络的输入进行相应的调整
    # 需要同时调整训练集的验证集
    # 主要是Peak_DNet网络输入的数据维度不同
    train_data = np.transpose(train_data, (0, 2, 1))    # [? , len_segment, 39]
    # train_data = np.expand_dims(train_data, axis=3)     # [?, 39, len_segment, 1]
    valid_data = np.transpose(valid_data, (0, 2, 1))
    # valid_data = np.expand_dims(valid_data, axis=3)
    print('Finished !!')

    # 使用了交叉熵损失函数
    # 未来可以尝试diceloss
    model.summary()
    model.compile(optimizer=Adam(lr=learning_rate, decay=learning_rate_decay), loss='binary_crossentropy',
                  metrics=list_metrics)
    # callback里面加入了自己写的peak_metrics
    peak_metrics = junxian_metrics.peak_callback(validation_data=(valid_data, valid_peeks))
    callback_list = [peak_metrics]
    his_list = []  # 保存每一次epoch的history
    list_pred_cv.append(list())  # 保存每一次epoch的pred
    # 自己写的epoch循环
    for epoch in range(epochs):
        print('Epoch %03d/%03d' % (epoch, epochs))
        time_start = time.time()
        history = model.fit(train_data, train_peeks, validation_data=(valid_data, valid_peeks),
                            epochs=1, batch_size=batch_size,
                            verbose=0, callbacks=callback_list, shuffle=True)
        time_train = time.time() - time_start
        dict_history = history.history
        print('Training time: %.2fs' % time_train)
        print('Train: loss: %.5f, ' % dict_history['loss'][0] +
              'Acc: %.4f, ' % dict_history['accuracy'][0] +
              'Catg_acc: %.4f, ' % dict_history['categorical_accuracy'][0] +
              'Recall: %.4f, ' % dict_history['recall'][0] +
              'Prcs: %.4f, ' % dict_history['precision'][0] +
              'F1: %.4f, ' % dict_history['f1_score'][0] +
              'Dice: %.4f, ' % dict_history['dice'][0])
        print('Valid: loss: %.5f, ' % dict_history['val_loss'][0] +
              'Acc: %.4f, ' % dict_history['val_accuracy'][0] +
              'Catg_acc: %.4f, ' % dict_history['val_categorical_accuracy'][0] +
              'Recall: %.4f, ' % dict_history['val_recall'][0] +
              'Prcs: %.4f, ' % dict_history['val_precision'][0] +
              'F1: %.4f, ' % dict_history['val_f1_score'][0] +
              'Dice: %.4f' % dict_history['val_dice'][0])

        print('       Peak Bias: Mean: %.4f, ' % peak_metrics.metrcis_dict['bias_mean'] +
              'Std: %.4f, ' % peak_metrics.metrcis_dict['bias_std'] +
              'Max: %03d, ' % peak_metrics.metrcis_dict['bias_max'] +
              'Min: %03d' % peak_metrics.metrcis_dict['bias_min'])
        dict_history['bias_arr'] = peak_metrics.metrcis_dict ['bias_arr']
        model.save(os.path.join(path_models, 'cv%d_epoch_%02d.h5' % (icv, epoch)))
        his_list.append(dict_history)
        y_pred = model.predict(valid_data, verbose=0)
        list_pred_cv[icv].append(y_pred)
    list_out.append(his_list)



bias_arr = 0
min_peak_bias_f1 = 0
min_epoch = 0
# 以下是根据每个epoch的指标，寻找结果最好的epoch
# 指标则根据平均值和标准差共同计算，
# 分别使用以下公式计算平均值和标准差的归一化指标
# 1- x / 100，得到一个范围基本上在(0,1]的值
# 根据F1-score的计算方式，计算平均值和标准差的调和平均数
for epoch in range(epochs):
    bias_arr2 = np.empty((0,))
    bias_f1 = 0
    for cv in range(5):
        bias_arr2 = np.concatenate((bias_arr2, list_out[cv][epoch]['bias_arr']), axis=0)
        bias_mean = 1 - np.abs(bias_arr2).mean() / 100
        bias_std = 1 - np.abs(bias_arr2).std() / 100
        bias_f1 = 2 * bias_mean * bias_std / (bias_mean + bias_std)  # 调和平均数
    if min_peak_bias_f1 < bias_f1:
        bias_arr = bias_arr2
        min_peak_bias_f1 = bias_f1
        min_epoch = epoch  # 找到调和平均数最小的epoch
list_out.append(min_epoch)

# 选取最小的epoch，得到那个epoch的网络在验证机的预测结果
bias_arr_pred = np.empty((0, len_segment))
for cv in list_pred_cv:
    bias_arr_pred = np.concatenate((bias_arr_pred, cv[min_epoch]), axis=0)

# 选取最小的epoch，保留那个网络的模型参数，删除其他的网络模型参数
list_model = os.listdir(path_models)
for h5 in list_model:
    if 'epoch_%02d.h5' % min_epoch not in h5:
        os.remove(os.path.join(path_models, h5))

# 保存相关结果
with open(os.path.join(path_outputs, 'list_out_cv.plk'), 'wb') as wf:
    pickle.dump(list_out, wf)
with open(os.path.join(path_outputs, 'list_out_cv.plk'), 'rb') as rf:
    try:
        print(pickle.load(rf))
    except EOFError:
        print(None)

with open(os.path.join(path_outputs, 'bias_arr.plk'), 'wb') as wf:
    pickle.dump(bias_arr, wf)
with open(os.path.join(path_outputs, 'bias_arr.plk'), 'rb') as rf:
    try:
        print(pickle.load(rf))
    except EOFError:
        print(None)

with open(os.path.join(path_outputs, 'bias_arr_pred.plk'), 'wb') as wf:
    pickle.dump(bias_arr_pred, wf)
with open(os.path.join(path_outputs, 'bias_arr_pred.plk'), 'rb') as rf:
    try:
        print(pickle.load(rf))
    except EOFError:
        print(None)

# 从临时文件夹中移出
shutil.move(path_log, os.path.join(path_output, time_now))

