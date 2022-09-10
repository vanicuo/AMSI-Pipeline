# 2020.11.12

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

sub_names = ['sub01', 'sub02', 'sub03', 'sub04', 'sub05',
             'sub06', 'sub07', 'sub08', 'sub09', 'sub10',
             'sub11', 'sub12', 'sub13', 'sub14', 'sub15',
             'sub16', 'sub17', 'sub18', 'sub19', 'sub20',
             'sub21', 'sub22', 'sub23', 'sub24', 'sub25',
             'sub26', 'sub27', 'sub28', 'sub29', 'sub30',
             'sub31', 'sub32', 'sub33', 'sub34', 'sub35']
num_case = len(sub_names)
date_exp = '2021_07_12_17_01_43'
path_output = os.path.join('../logs/Peak_dnet_junxian',
                           date_exp, 'output')

with open(os.path.join(path_output, 'bias_arr.plk'), 'rb') as f:
    bias_arr = pickle.load(f)
with open(os.path.join(path_output, 'bias_arr_pred.plk'), 'rb') as f:
    bias_arr_pred2 = pickle.load(f)

len_data = len(bias_arr)
len_segment = bias_arr_pred2.shape[1]
if len_segment != 300:
    bias_arr_pred = np.zeros((len_data, 300))
    bias_arr_pred[:, 2:-2] = bias_arr_pred2
else:
    bias_arr_pred = bias_arr_pred2

len_data2 = round(len_data/10)
bias_arr2 = np.zeros((len_data2,))
for i in range(len_data2):
    bias_arr_tmp = bias_arr[i*10:i*10+10, ]
    bias_arr2[i] = np.abs(bias_arr_tmp).min()


print('Loading Dataset')
with open('../preprocess/meg_train_ori.pkl', 'rb') as f:
    data_meg = pickle.load(f)
with open('../preprocess/train_peeks_sgm010.pkl', 'rb') as f:
    data_peeks = pickle.load(f)

data = np.empty((0, 39, 300, 1))
peeks = np.empty((0, 300))
num_cv = 7
num_valid = int(num_case / num_cv)

for icv in range(num_cv):
    valid_index = range(icv * num_valid, (icv + 1) * num_valid)
    for cases in valid_index:
        data = np.concatenate((data, data_meg[cases]), axis=0)
        peeks = np.concatenate((peeks, data_peeks[cases]), axis=0)

list_bias_arr = []
list_bias_arr_pred = []
sum_len_data_case = 0
for case in range(num_case):
    len_data_case = data_peeks[case].shape[0]
    list_bias_arr.append(bias_arr[sum_len_data_case: sum_len_data_case + len_data_case])
    list_bias_arr_pred.append(bias_arr_pred[sum_len_data_case: sum_len_data_case + len_data_case, :])
    sum_len_data_case += len_data_case

t = np.arange(0, 300, 1)
# path_fig = os.path.join('../output/meg_peek_fig/', date_exp)
path_fig = os.path.join('/data2/zhengli/peak_detection/output/meg_peak_fig/', date_exp)
if not os.path.exists(path_fig):
    os.makedirs(path_fig)
for case in range(num_case):
    case_name = sub_names[case]
    print(case_name)
    path_fig_case = os.path.join(path_fig, case_name)
    if not os.path.exists(path_fig_case):
        os.mkdir(path_fig_case)
    len_data_case = data_peeks[case].shape[0]
    for seg in range(len_data_case):
        fig = plt.figure(figsize=(8, 8))
        seg_data_ = data_meg[case][seg, :, :, 0]  # [39, 300]
        seg_data_grads = data_meg[case][seg, 0:26, :, 0]  # [27, 300]
        seg_data_mags = data_meg[case][seg, 26:39, :, 0]  # [12, 300]

        plt.subplot(411)
        plt.plot(t, seg_data_grads.T)
        plt.ylabel('GRADS')
        title = case_name + '_peak%04d' % seg + '_bias%03d' % list_bias_arr[case][seg]
        plt.title(title)

        plt.subplot(412)
        plt.plot(t, seg_data_mags.T)
        plt.ylabel('MAGS')

        plt.subplot(414)
        peek_data = data_peeks[case][seg,:]  # [300]
        peek_data_pred300 = list_bias_arr_pred[case][seg, :]  # [300, 1]
        peek_data_pred300 = (peek_data_pred300 - peek_data_pred300.min()) / (peek_data_pred300.max() - peek_data_pred300.min())
        bias = int(list_bias_arr[case][seg])
        peek_data_pred = np.zeros(300)
        if bias > 0:
            peek_data_pred[bias:] = peek_data[0:-bias]
        elif bias < 0:
            peek_data_pred[:bias] = peek_data[-bias:]
        else:
            peek_data_pred = peek_data

        # peek_data = np.concatenate([peek_data[:, np.newaxis], peek_data_pred[:, np.newaxis]], axis=1)
        peek_data = np.concatenate([peek_data_pred[:, np.newaxis], peek_data[:, np.newaxis]], axis=1)
        plt.plot(t, peek_data)
        plt.ylabel('PEEK')

        plt.subplot(413)
        plt.plot(t, peek_data_pred300)
        plt.xlabel('Times (ms)')
        plt.ylabel('PRED')
        plt.subplots_adjust()
        file_save = 'bias%03d' % abs(list_bias_arr[case][seg]) + '_peak%04d' % seg + '.jpg'
        print(file_save)
        plt.savefig(os.path.join(path_fig_case, file_save))
        # plt.show()
        plt.close('all')
        del fig



