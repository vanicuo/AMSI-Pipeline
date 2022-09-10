# coding=utf-8
# cmr
# 用于测试网络是否能跑通

import numpy as np
from keras.optimizers import *
import junxian_lib.peak_metrics as junxian_metrics
import net.unet_1D as unet
from metrics import recall, precision, f1_score
learning_rate = 0.001
learning_rate_decay = 0.0001
batch_size = 64
list_metrics = ['accuracy', 'categorical_accuracy', recall, precision, f1_score, junxian_metrics.dice()]
list_metrics_index = ['accuracy', 'categorical_accuracy', 'recall', 'precision', 'f1_score', 'loss']


net = unet.unet_1D_v4((296, 39))
# net = Peak_DNet_junxian((39, 300, 1))
net.compile(optimizer=Adam(lr=learning_rate, decay=learning_rate_decay), loss='binary_crossentropy',
              metrics=list_metrics)
train_data = np.ones((256,296, 39))
train_peeks = np.ones((256,296))
history = net.fit(train_data, train_peeks, epochs=1, batch_size=batch_size,
                    verbose=0, shuffle=True)