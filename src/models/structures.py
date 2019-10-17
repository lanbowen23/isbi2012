from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, BatchNormalization, Activation, Reshape, Dropout
from keras.layers.advanced_activations import PReLU
from keras.layers import concatenate

def conv_block(inputs, channel, down_size=True, drop_out=False):
    conv = Conv2D(channel, (3, 3), padding='same', kernel_initializer='he_normal')(inputs)
    # conv = Activation('relu')(conv)
    conv = PReLU(shared_axes=[1,2])(conv)
    conv = Conv2D(channel, (3, 3), padding='same', kernel_initializer='he_normal')(conv)
    # conv = Activation('relu')(conv)
    conv = PReLU(shared_axes=[1,2])(conv)
    if drop_out:
        conv = Dropout(0.5)(conv)
    if down_size:
        return conv, MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv)
    return conv

def conv_block_BN(inputs, channel, down_size=True, drop_out=False):
    conv = Conv2D(channel, (3, 3), padding='same', kernel_initializer='he_normal')(inputs)
    # conv = Activation('relu')(conv)
    conv = PReLU(shared_axes=[1,2])(conv)
    conv = BatchNormalization()(conv)
    conv = Conv2D(channel, (3, 3), padding='same', kernel_initializer='he_normal')(conv)
    # conv = Activation('relu')(conv)
    conv = PReLU(shared_axes=[1,2])(conv)
    conv = BatchNormalization()(conv)
    if drop_out:
        conv = Dropout(0.5)(conv)
    if down_size:
        return conv, MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv)
    return conv

def conv_block_aspp(inputs, channel, down_size=True, drop_out=False):
    conv = Conv2D(channel, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    c1 = Conv2D(int(channel/2), (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv)
    c2 = Conv2D(int(channel/2), (3, 3), dilation_rate=6, activation='relu', padding='same', kernel_initializer='he_normal')(conv)
    c3 = Conv2D(int(channel/2), (3, 3), dilation_rate=12, activation='relu', padding='same', kernel_initializer='he_normal')(conv)
    c4 = Conv2D(int(channel/2), (3, 3), dilation_rate=18, activation='relu', padding='same', kernel_initializer='he_normal')(conv)
    conv = concatenate([c1, c2, c3, c4], axis=3)
    conv = BatchNormalization()(conv)
    conv = Conv2D(channel, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal')(conv)
    if drop_out:
        conv = Dropout(0.5)(conv)
    if down_size:
        return conv, MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv)
    return conv


def unet3(inputs, down_depth=32, up_depth=256, drop=True):
    conv1, pool1 = conv_block(inputs, down_depth, down_size=True)
    conv2, pool2 = conv_block(pool1, int(down_depth*2), down_size=True)
    conv3, pool3 = conv_block(pool2, int(down_depth*4), down_size=True, drop_out=drop)
    conv4 = conv_block(pool3, int(down_depth*8), down_size=False, drop_out=drop)

    up5 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv3], axis=3)
    conv5 = conv_block(up5, up_depth, down_size=False, drop_out=drop)
    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv2], axis=3)
    conv6 = conv_block(up6, int(up_depth/2), down_size=False)
    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv1], axis=3)
    conv7 = conv_block(up7, int(up_depth/4), down_size=False)
    return conv7

def unet3_deconv(inputs, down_depth=32, up_depth=256, drop=False):
    conv1, pool1 = conv_block(inputs, down_depth, down_size=True)
    conv2, pool2 = conv_block(pool1, int(down_depth*2), down_size=True)
    conv3, pool3 = conv_block(pool2, int(down_depth*4), down_size=True, drop_out=drop)
    conv4 = conv_block(pool3, int(down_depth*8), down_size=False, drop_out=drop)

    conv4 = Conv2DTranspose(up_depth, kernel_size=(3, 3), strides=(2, 2), padding="same", kernel_initializer='he_normal')(conv4)
    up5 = concatenate([conv4, conv3], axis=3)
    conv5 = conv_block(up5, up_depth, down_size=False, drop_out=drop)
    conv5 = Conv2DTranspose(int(up_depth/2), kernel_size=(3, 3), strides=(2, 2), padding="same", kernel_initializer='he_normal')(conv5)
    up6 = concatenate([conv5, conv2], axis=3)
    conv6 = conv_block(up6, int(up_depth/2), down_size=False)
    conv6 = Conv2DTranspose(int(up_depth/4), kernel_size=(3, 3), strides=(2, 2), padding="same", kernel_initializer='he_normal')(conv6)    
    up7 = concatenate([conv6, conv1], axis=3)
    conv7 = conv_block(up7, int(up_depth/4), down_size=False)
    return conv7

def unet3_BN(inputs, down_depth=32, up_depth=256, drop=True):
    conv1, pool1 = conv_block_BN(inputs, down_depth, down_size=True)
    conv2, pool2 = conv_block_BN(pool1, int(down_depth*2), down_size=True)
    conv3, pool3 = conv_block_BN(pool2, int(down_depth*4), down_size=True, drop_out=drop)
    conv4 = conv_block_BN(pool3, int(down_depth*8), down_size=False, drop_out=drop)

    up5 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv3], axis=3)
    conv5 = conv_block_BN(up5, up_depth, down_size=False, drop_out=drop)
    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv2], axis=3)
    conv6 = conv_block_BN(up6, int(up_depth/2), down_size=False)
    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv1], axis=3)
    conv7 = conv_block_BN(up7, int(up_depth/4), down_size=False)
    return conv7

def unet3_aspp(inputs, down_depth=32, up_depth=256, dropout5=False):
    conv1, pool1 = conv_block_aspp(inputs, down_depth, down_size=True)
    conv2, pool2 = conv_block_aspp(pool1, int(down_depth*2), down_size=True, drop_out=dropout5)
    conv3, pool3 = conv_block_aspp(pool2, int(down_depth*4), down_size=True, drop_out=True)
    conv4 = conv_block_aspp(pool3, int(down_depth*8), down_size=False, drop_out=True)

    up5 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv3], axis=3)
    conv5 = conv_block(up5, up_depth, down_size=False, drop_out=True)
    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv2], axis=3)
    conv6 = conv_block(up6, int(up_depth/2), down_size=False, drop_out=dropout5)
    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv1], axis=3)
    conv7 = conv_block(up7, int(up_depth/4), down_size=False)
    return conv7

def unet4(inputs):
    conv1, pool1 = conv_block(inputs, 32, down_size=True)
    conv2, pool2 = conv_block(pool1, 64, down_size=True)
    conv3, pool3 = conv_block(pool2, 128, down_size=True, drop_out=True)
    conv8, pool8 = conv_block(pool3, 256, down_size=True, drop_out=True)
    conv4 = conv_block(pool8, 512, down_size=False, drop_out=True)

    up5 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv8], axis=3)
    conv5 = conv_block(up5, 256, down_size=False, drop_out=True)
    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv3], axis=3)
    conv6 = conv_block(up6, 128, down_size=False, drop_out=True)
    up9 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv2], axis=3)
    conv9 = conv_block(up9, 64, down_size=False)
    up7 = concatenate([UpSampling2D(size=(2, 2))(conv9), conv1], axis=3)
    conv7 = conv_block(up7, 32, down_size=False) 

    return conv7