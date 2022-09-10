# coding=utf-8
# CMR
# print每一次实验的指标，方便查看
# /logs/Peak_dnet_junxian为实验结果保存路径

import os
import pickle
import numpy as np

sub_names = ['sub01', 'sub02', 'sub03', 'sub04', 'sub05',
             'sub06', 'sub07', 'sub08', 'sub09', 'sub10',
             'sub11', 'sub12', 'sub13', 'sub14', 'sub15',
             'sub16', 'sub17', 'sub18', 'sub19', 'sub20',
             'sub21', 'sub22', 'sub23', 'sub24', 'sub25',
             'sub26', 'sub27', 'sub28', 'sub29', 'sub30',
             'sub31', 'sub32', 'sub33', 'sub34', 'sub35']
num_case = len(sub_names)
path_log = '../logs/Peak_dnet_junxian'
list_exp = os.listdir(path_log)
list_exp.remove('temp')

num_exp = len(list_exp)
mean_arr = np.zeros(num_exp)
std_arr = np.zeros(num_exp)
mean10_arr = np.zeros(num_exp)
std10_arr = np.zeros(num_exp)


for j in range(num_exp):
    exper = list_exp[j]
    path_output = os.path.join(path_log, exper, 'output')
    with open(os.path.join(path_output, 'bias_arr.plk'), 'rb') as f:
        bias_arr = pickle.load(f)
    with open(os.path.join(path_output, 'list_out_cv.plk'), 'rb') as f:
        list_out = pickle.load(f)


    len_data = len(bias_arr)
    len_data2 = round(len_data / 10)
    bias_arr2 = np.zeros((len_data2,))
    for i in range(len_data2):
        bias_arr_tmp = bias_arr[i*10:i*10+10, ]
        bias_arr2[i] = np.abs(bias_arr_tmp).min()

    print(exper)
    mean_arr[j] = np.abs(bias_arr).mean()
    std_arr[j] = np.abs(bias_arr).std()
    mean10_arr[j] = np.abs(bias_arr2).mean()
    std10_arr[j] = np.abs(bias_arr2).mean()
    print(list_out[-1])
    print('mean:', np.abs(bias_arr).mean())
    print('STD:', np.abs(bias_arr).std())
    print('mean2:', np.abs(bias_arr2).mean())
    print('STD2:', np.abs(bias_arr2).std())