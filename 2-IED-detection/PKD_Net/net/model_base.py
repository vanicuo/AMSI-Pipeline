# coding=utf-8
# 2020.11.03
# CMR
# 主要包含一些网络基础模块
# PS：decoder与encoder命名时反了

from keras.models import Model
import keras.layers as layers


# 最基础的卷积+激活+BN层，两层连在一起，depth为U-net的深度，base_feature为feature map的数量倍数，res为是否使用resblock
def conv_block(inputs, depth=1, padding="valid", data_format='channels_first', base_featrue=16, kernel_size=3, res=False):  # (x, ch)
    if res:
        padding = 'same'
        inputs = layers.Conv1D(base_featrue * depth,
                          kernel_size=1,
                          padding=padding,
                          data_format=data_format)(inputs)  # (x-2, 16*depth)
        inputs = layers.LeakyReLU(alpha=0.1)(inputs)
        inputs = layers.BatchNormalization()(inputs)

    x = layers.Conv1D(base_featrue * depth,
                      kernel_size=kernel_size,
                      padding=padding,
                      data_format=data_format)(inputs)  # (x-2, 16*depth)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(base_featrue * depth,
                      kernel_size=kernel_size,
                      padding=padding,
                      data_format=data_format)(x)  # (x-4, 16*depth)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.BatchNormalization()(x)
    if res:
        x = layers.Add()([inputs, x])
    return x

# 单通道的decoder
def unet_decoder_channel(inputs_size=(1,296), depth=1, kernel_size=3, base_feature=2):
    inputs = layers.Input(shape=inputs_size)
    depth = depth
    data_format = 'channels_first'

    skip_con = conv_block(inputs, depth, 'same', data_format, base_featrue=base_feature, kernel_size=kernel_size)  # 296 * (1*base_feature)
    base_model = Model(inputs=inputs, outputs=skip_con)
    return base_model

# 降采样层
def downsampling(skip_con, data_format='channels_first'):
    x = layers.Reshape((int(skip_con.shape[1]) * int(skip_con.shape[2]), int(skip_con.shape[3])))(skip_con)  # [?, 78, 296]
    x = layers.AvgPool1D(data_format=data_format)(x)  # [?, 78, 148]
    x = layers.Reshape((int(skip_con.shape[1]), int(skip_con.shape[2]), int(x.shape[2])))(x)  # [?, 39, 2, 148]
    return x

# 将单通道的decoder与降采样放在一起
# 可以选择不同的数据降维方法，注释不同的代码段即可
def unet_decoder_block(x, depth, kernel_size, base_feature, data_format):
    x_size = (int(x.shape[2]), int(x.shape[3]))
    decoder = unet_decoder_channel(inputs_size=x_size, depth=depth, kernel_size=kernel_size, base_feature=base_feature)
    skip_con = layers.TimeDistributed(decoder)(x)  # [?, 39, 2, 296]
    x = downsampling(skip_con, data_format=data_format)  # [?, 39, 2, 148]

    ## 直接展开
    skip_con = layers.Reshape((int(skip_con.shape[1]) * int(skip_con.shape[2]), int(skip_con.shape[3])))(skip_con) # [?, 39* 2, 296]

    # 在feature map维度上进行global avg
    # skip_con = layers.Permute((1, 3, 2))(skip_con) # [?, 39, 296, 2]
    # ch1 = int(skip_con.shape[1])
    # ch2 = int(skip_con.shape[2])
    # skip_con = layers.Reshape((ch1 * ch2, int(skip_con.shape[3])))(skip_con) # [?, 39*296, 2]
    # skip_con = layers.GlobalAveragePooling1D(data_format=data_format)(skip_con) # [?, 39*296]
    # skip_con = layers.Reshape((ch1, ch2))(skip_con) # [?, 39, 296]

    ## 在meg channel维度上进行global avg
    # skip_con = layers.Permute((2, 3, 1))(skip_con) # [?, 2, 296, 39,]
    # ch1 = int(skip_con.shape[1])
    # ch2 = int(skip_con.shape[2])
    # skip_con = layers.Reshape((ch1 * ch2, int(skip_con.shape[3])))(skip_con) # [?, 2*296, 39]
    # skip_con = layers.GlobalAveragePooling1D(data_format=data_format)(skip_con) # [?, 2*296]
    # skip_con = layers.Reshape((ch1, ch2))(skip_con) # [?, 2, 296]

    depth *= 2
    return x, skip_con, depth


# encoder block，将升采样和跳连接放在了一起，再加入了卷积模块
def unet_encoder_block(x, skip_con, depth, kernel_size, base_feature, data_format):
    x = layers.Permute((2,1))(x)
    x = layers.UpSampling1D()(x)  # [?, 39*16, 74]
    x = layers.Permute((2, 1))(x)
    x = layers.concatenate([x, skip_con], axis=1)  # [?, 39*24, 74]
    x = conv_block(x, depth, 'same', data_format, base_feature, kernel_size)  # [?, 128, 74]
    depth /= 2
    return x, depth