# coding=utf-8
# 2020.11.03
# CMR

import net.model_base as model_base
import keras.layers as layers
from keras.models import Model
import tensorflow as tf

# valid padding的U-Net
def unet_1D(input_size=(300, 39), data_format='channels_last', base_feature=16):
    inputs = layers.Input(shape=input_size)
    depth = 1
    skip_con01 = model_base.conv_block(inputs, depth, data_format=data_format, base_featrue=base_feature)   # 296 * (1*base_feature)
    x = layers.AvgPool1D(pool_size=2, data_format=data_format)(skip_con01)   # 148
    depth *= 2
    skip_con02 = model_base.conv_block(x, depth, data_format=data_format, base_featrue=base_feature)    # 144 * (2*base_feature)
    x = layers.AvgPool1D(pool_size=2, data_format=data_format)(skip_con02)   # 72
    depth *= 2
    skip_con03 = model_base.conv_block(x, depth, data_format=data_format, base_featrue=base_feature)    # 68 * (4*base_feature)
    x = layers.AvgPool1D(pool_size=2, data_format=data_format)(skip_con03)   # 34
    depth *= 2
    x = model_base.conv_block(x, depth, data_format=data_format, base_featrue=base_feature)    # 30 * (8*base_feature)
    x = layers.UpSampling1D(2)(x)                                       # 60
    len_skip_con_crop = int((int(skip_con03.shape[1]) - int(x.shape[1]))/2)
    skip_con03 = layers.Cropping1D(len_skip_con_crop)(skip_con03)
    x = layers.concatenate([x, skip_con03], axis=2)     # 60 * 192
    depth /= 2
    x = model_base.conv_block(x, depth, data_format=data_format, base_featrue=base_feature)    # 56 * (4*base_feature)
    x = layers.UpSampling1D(2)(x)  # 112 * 64
    len_skip_con_crop = int((int(skip_con02.shape[1]) - int(x.shape[1])) / 2)
    skip_con02 = layers.Cropping1D(len_skip_con_crop)(skip_con02)
    x = layers.concatenate([x, skip_con02], axis=2)     # 112 * 96
    depth /= 2
    x = model_base.conv_block(x, depth, data_format=data_format, base_featrue=base_feature)  # 108 * (2*base_feature)
    x = layers.UpSampling1D(2)(x)  # 216 * 32
    len_skip_con_crop = int((int(skip_con01.shape[1]) - int(x.shape[1])) / 2)
    skip_con01 = layers.Cropping1D(len_skip_con_crop)(skip_con01)
    x = layers.concatenate([x, skip_con01], axis=2)     # 216 * 48
    depth /= 2
    x = model_base.conv_block(x, depth, data_format=data_format, base_featrue=base_feature)  # 212 * (2*base_feature)
    x = layers.Conv1D(1, kernel_size=1, padding='valid', data_format=data_format)(x)    # 212 * 1
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Flatten()(x)
    out = layers.Dense(300, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=out)
    return model

# same padding的U-Net, kernel_size统一
def unet_1D_v2(input_size=(296, 39), data_format='channels_last', kernel_size=3, base_feature=16):
    inputs = layers.Input(shape=input_size)
    depth = 1
    skip_con01 = model_base.conv_block(inputs, depth, 'same', data_format, base_featrue=base_feature, kernel_size=kernel_size)   # 296 * (1*base_feature)
    x = layers.AvgPool1D(pool_size=2, data_format=data_format)(skip_con01)   # 148
    depth *= 2
    skip_con02 = model_base.conv_block(x, depth, 'same', data_format, base_featrue=base_feature, kernel_size=kernel_size)    # 148 * (2*base_feature)
    x = layers.AvgPool1D(pool_size=2, data_format=data_format)(skip_con02)   # 74
    depth *= 2
    skip_con03 = model_base.conv_block(x, depth, 'same', data_format, base_featrue=base_feature, kernel_size=kernel_size)    # 74 * (4*base_feature)
    x = layers.AvgPool1D(pool_size=2, data_format=data_format)(skip_con03)   # 37
    depth *= 2
    x = model_base.conv_block(x, depth, 'same', data_format, base_featrue=base_feature, kernel_size=kernel_size)    # 37 * (8*base_feature)
    x = layers.UpSampling1D(2)(x)                       # 74
    x = layers.concatenate([x, skip_con03], axis=2)     # 74 * (12*base_feature)
    depth /= 2
    x = model_base.conv_block(x, depth, 'same', data_format, base_featrue=base_feature, kernel_size=kernel_size)    # 74 * (4*base_feature)
    x = layers.UpSampling1D(2)(x)  # 112 * (4*base_feature)
    x = layers.concatenate([x, skip_con02], axis=2)     # 148 * (6*base_feature)
    depth /= 2
    x = model_base.conv_block(x, depth, 'same', data_format, base_featrue=base_feature, kernel_size=kernel_size)  # 148 * (2*base_feature)
    x = layers.UpSampling1D(2)(x)  # 216 * (2*base_feature)
    x = layers.concatenate([x, skip_con01], axis=2)     # 296 * (3*base_feature)
    depth /= 2
    x = model_base.conv_block(x, depth, 'same', data_format, base_featrue=base_feature, kernel_size=kernel_size)  # 296 * (base_feature)
    out = layers.Conv1D(1, kernel_size=1, activation='sigmoid', data_format=data_format)(x)    # 296 * 1
    out = layers.Reshape((296,))(out)
    model = Model(inputs=inputs, outputs=out)
    return model

# 减少了一次降采样与升采样次数的U-Net，预实验效果不好，没有尝试
def unet_1D_v3(input_size=(296, 39), data_format='channels_last', kernel_size=5, base_feature=16):
    inputs = layers.Input(shape=input_size)
    depth = 1
    skip_con01 = model_base.conv_block(inputs, depth, 'same', data_format,
                                       base_featrue=base_feature, kernel_size=kernel_size)   # 296 * (1*base_feature)
    x = layers.AvgPool1D(pool_size=2, data_format=data_format)(skip_con01)   # 148
    depth *= 2
    skip_con02 = model_base.conv_block(x, depth, 'same', data_format,
                                       base_featrue=base_feature, kernel_size=kernel_size)   # 148 * (2*base_feature)
    x = layers.AvgPool1D(pool_size=2, data_format=data_format)(skip_con02)   # 74
    depth *= 2
    x = model_base.conv_block(x, depth, 'same', data_format,
                              base_featrue=base_feature, kernel_size=kernel_size)   # 74 * (2*base_feature)
    x = layers.UpSampling1D(2)(x)  # 112 * (4*base_feature)
    x = layers.concatenate([x, skip_con02], axis=2)     # 148 * (6*base_feature)
    depth /= 2
    x = model_base.conv_block(x, depth, 'same', data_format,
                              base_featrue=base_feature, kernel_size=kernel_size)   # 148 * (2*base_feature)
    x = layers.UpSampling1D(2)(x)  # 148 * (2*base_feature)
    x = layers.concatenate([x, skip_con01], axis=2)     # 296 * (3*base_feature)
    depth /= 2
    x = model_base.conv_block(x, depth, 'same', data_format,
                              base_featrue=base_feature, kernel_size=kernel_size)   # 296 * (2*base_feature)
    out = layers.Conv1D(1, kernel_size=1, activation='sigmoid', data_format=data_format)(x)    # 296 * 1
    model = Model(inputs=inputs, outputs=out)
    return model


# 不同的encoder与decoder的U-Net
def unet_1D_v4(inputs_size=(296, 39), kernel_size=3, base_feature=2, data_format='channels_first'):
    inputs_raw = layers.Input(shape=inputs_size)
    inputs = layers.Permute((2, 1))(inputs_raw)         # (39, 296)
    inputs = layers.Reshape((39, 296, 1))(inputs)   # (39, 296, 1)
    inputs = layers.Permute((1, 3, 2))(inputs)      # (39, 1, 296)
    depth = 1
    x, skip_con01, depth = model_base.unet_decoder_block(inputs, depth, kernel_size, base_feature, data_format) # x:[?, 39, 2, 148], skip_con01:[?, 39*2, 296]
    x, skip_con02, depth = model_base.unet_decoder_block(x, depth, kernel_size, base_feature, data_format) # x:[?, 39, 4, 74] , skip_con02:[?, 39*4, 148]
    x, skip_con03, depth = model_base.unet_decoder_block(x, depth, kernel_size, base_feature, data_format) # x:[?, 39, 8, 37] , skip_con03:[?, 39*8, 74]

    decoder = model_base.unet_decoder_channel(inputs_size=(int(x.shape[2]), int(x.shape[3])), depth=depth,
                                             kernel_size=kernel_size, base_feature=base_feature)
    x = layers.TimeDistributed(decoder)(x)  # [?, 39, 16, 37]
    x = layers.Reshape((int(x.shape[1]) * int(x.shape[2]), int(x.shape[3])))(x)  # [?, 39*16, 37]

    base_feature *= 8
    x, depth = model_base.unet_encoder_block(x, skip_con03, depth, kernel_size, base_feature, data_format) # [?, 128, 74]
    x, depth = model_base.unet_encoder_block(x, skip_con02, depth, kernel_size, base_feature, data_format) # [?, 64, 148]
    x, depth = model_base.unet_encoder_block(x, skip_con01, depth, kernel_size, base_feature, data_format) # [?, 32, 296]
    x = model_base.conv_block(x, depth, 'same', data_format, base_featrue=base_feature, kernel_size=kernel_size)  # [?, 16, 296]
    out = layers.Conv1D(1, kernel_size=1, activation='sigmoid', data_format=data_format)(x)    # 296 * 1
    # out = layers.Permute((2,1))(out)
    out = layers.Reshape((296,))(out)
    model = Model(inputs=inputs_raw, outputs=out)
    # x_size = (x[1], x[2])
    # decoder1 = model_base.unet_decoder_channel(inputs_size=x_size, depth=depth,
    #                                          kernel_size=kernel_size, base_feature=base_feature)
    # skip_con01 = layers.TimeDistributed(decoder1)(x)  # [?, 39, 2, 296]
    # x = model_base.downsampling(skip_con01, data_format=data_format)  # [?, 39, 2, 148]

    # x = layers.Reshape((int(skip_con01.shape[1]) * int(skip_con01.shape[2]), int(skip_con01.shape[3])))(skip_con01)  # [?, 78, 296]
    # x = layers.AvgPool1D(data_format=data_format)(x)  # [?, 78, 148]
    # x = layers.Reshape((int(skip_con01.shape[1]), int(skip_con01.shape[2]), int(x.shape[2])))(x)  # [?, 39, 2, 148]
    # depth *= 2
    # x_size = (x[1], x[2])
    # decoder2 = model_base.unet_decoder_channel(inputs_size=x_size, depth=depth,
    #                                          kernel_size=kernel_size, base_feature=base_feature)
    # skip_con02 = layers.TimeDistributed(decoder2)(x)  # [?, 39, 4, 148]
    # depth *= 2
    # x = model_base.downsampling(skip_con02, data_format=data_format)  # [?, 39, 4, 74]
    #
    # x_size = (x[1], x[2])
    # decoder3 = model_base.unet_decoder_channel(inputs_size=x_size, depth=depth,
    #                                          kernel_size=kernel_size, base_feature=base_feature)
    # skip_con03 = layers.TimeDistributed(decoder3)(x)  # [?, 39, 8, 74]
    # x = model_base.downsampling(skip_con03, data_format=data_format)  # [?, 39, 8, 37]

    return model

# 需要自己定义各级的kernel_size
def unet_1D_v5(input_size=(296, 39), data_format='channels_last', base_feature=16):
    inputs = layers.Input(shape=input_size)
    depth = 1
    skip_con01 = model_base.conv_block(inputs, depth, 'same', data_format, base_featrue=base_feature, kernel_size=40)   # 296 * (1*base_feature)
    x = layers.MaxPool1D(pool_size=2, data_format=data_format)(skip_con01)   # 148
    depth *= 2
    skip_con02 = model_base.conv_block(x, depth, 'same', data_format, base_featrue=base_feature, kernel_size=30)    # 148 * (2*base_feature)
    x = layers.MaxPool1D(pool_size=2, data_format=data_format)(skip_con02)   # 74
    depth *= 2
    skip_con03 = model_base.conv_block(x, depth, 'same', data_format, base_featrue=base_feature, kernel_size=20)    # 74 * (4*base_feature)
    x = layers.MaxPool1D(pool_size=2, data_format=data_format)(skip_con03)   # 37
    depth *= 2
    x = model_base.conv_block(x, depth, 'same', data_format, base_featrue=base_feature, kernel_size=10)    # 37 * (8*base_feature)
    x = layers.UpSampling1D(2)(x)                       # 74
    x = layers.concatenate([x, skip_con03], axis=2)     # 74 * (12*base_feature)
    depth /= 2
    x = model_base.conv_block(x, depth, 'same', data_format, base_featrue=base_feature, kernel_size=20)    # 74 * (4*base_feature)
    x = layers.UpSampling1D(2)(x)  # 112 * (4*base_feature)
    x = layers.concatenate([x, skip_con02], axis=2)     # 148 * (6*base_feature)
    depth /= 2
    x = model_base.conv_block(x, depth, 'same', data_format, base_featrue=base_feature, kernel_size=30)  # 148 * (2*base_feature)
    x = layers.UpSampling1D(2)(x)  # 216 * (2*base_feature)
    x = layers.concatenate([x, skip_con01], axis=2)     # 296 * (3*base_feature)
    depth /= 2
    x = model_base.conv_block(x, depth, 'same', data_format, base_featrue=base_feature, kernel_size=40)  # 296 * (base_feature)
    out = layers.Conv1D(1, kernel_size=1, activation='sigmoid', data_format=data_format)(x)    # 296 * 1
    out = layers.Reshape((296,))(out)
    model = Model(inputs=inputs, outputs=out)
    return model

# 加入了res_block的U-Net，同时也需要自己定义各级的kernel_size
def unet_1D_v6(input_size=(296, 39), data_format='channels_last', base_feature=16):
    inputs = layers.Input(shape=input_size)
    depth = 1
    skip_con01 = model_base.conv_block(inputs, depth, 'same', data_format, base_featrue=base_feature, kernel_size=40, res=True)   # 296 * (1*base_feature)
    x = layers.AvgPool1D(pool_size=2, data_format=data_format)(skip_con01)   # 148
    # x = layers.AvgPool1D(pool_size=2, data_format=data_format)(skip_con01)   # 148
    depth *= 2
    skip_con02 = model_base.conv_block(x, depth, 'same', data_format, base_featrue=base_feature, kernel_size=30, res=True)    # 148 * (2*base_feature)
    x = layers.AvgPool1D(pool_size=2, data_format=data_format)(skip_con02)   # 74
    depth *= 2
    skip_con03 = model_base.conv_block(x, depth, 'same', data_format, base_featrue=base_feature, kernel_size=20, res=True)    # 74 * (4*base_feature)
    x = layers.AvgPool1D(pool_size=2, data_format=data_format)(skip_con03)   # 37
    depth *= 2
    x = model_base.conv_block(x, depth, 'same', data_format, base_featrue=base_feature, kernel_size=10, res=True)    # 37 * (8*base_feature)
    x = layers.UpSampling1D(2)(x)                       # 74
    x = layers.concatenate([x, skip_con03], axis=2)     # 74 * (12*base_feature)
    depth /= 2
    x = model_base.conv_block(x, depth, 'same', data_format, base_featrue=base_feature, kernel_size=20, res=True)    # 74 * (4*base_feature)
    x = layers.UpSampling1D(2)(x)  # 112 * (4*base_feature)
    x = layers.concatenate([x, skip_con02], axis=2)     # 148 * (6*base_feature)
    depth /= 2
    x = model_base.conv_block(x, depth, 'same', data_format, base_featrue=base_feature, kernel_size=30, res=True)  # 148 * (2*base_feature)
    x = layers.UpSampling1D(2)(x)  # 216 * (2*base_feature)
    x = layers.concatenate([x, skip_con01], axis=2)     # 296 * (3*base_feature)
    depth /= 2
    x = model_base.conv_block(x, depth, 'same', data_format, base_featrue=base_feature, kernel_size=40, res=True)  # 296 * (base_feature)
    out = layers.Conv1D(1, kernel_size=1, activation='sigmoid', data_format=data_format)(x)    # 296 * 1
    out = layers.Reshape((296,))(out)
    model = Model(inputs=inputs, outputs=out)
    return model