

# -*- coding: utf_8 -*-

import os
import cv2
import pickle

import numpy as np
import keras
from keras.models import Sequential
from keras.callbacks import ReduceLROnPlateau
from keras import backend as K
from keras.engine.input_layer import Input
from keras.models import Model
from keras import regularizers
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Lambda, AveragePooling2D, GlobalAveragePooling2D, Activation, concatenate
from keras.regularizers import l2
from keras.utils import multi_gpu_model


"""
Model Architecture

- build_model()
"""

def build_resnet_pretrained(input_shape):

    from keras.applications.resnet50 import ResNet50

    model = ResNet50(input_shape=input_shape, include_top=False)
    x1 = GlobalAveragePooling2D()(model.output)
    x2 = Dense(1383, use_bias=True, bias_initializer='zeros')(x1)
    x3 = Activation('softmax')(x2)
    model_new = Model(inputs=model.input, outputs=x3)
    model_new.summary()

    return model_new

def build_model(input_shape):
    '''
    h1 : top of unet
    h2 : second of unet
    h3 : third of unet
    h4 : forth of unet
    '''
    inputs = Input(shape=input_shape)
    #224x224x3

    x = Conv2D(64, (3, 3), strides=1, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), strides=2, padding='same')(x)
    h1 = Activation('relu')(x)
    # 112x112x64

    ##### DENSE BLOCK 1 #####

    bn_1 = BatchNormalization()(max_pool)
    conv_1 = Conv2D(128, (1, 1), padding='same')(bn_1)
    act_1 = Activation('relu')(conv_1)
    bn_2 = BatchNormalization()(act_1)
    conv_2 = Conv2D(32, (3, 3), padding='same')(bn_2)
    act_2 = Activation('relu')(conv_2)
    merged_1 = concatenate([act_2, max_pool], axis=-1)

    bn_1 = BatchNormalization()(merged_1)
    conv_1 = Conv2D(128, (1, 1), padding='same')(bn_1)
    act_1 = Activation('relu')(conv_1)
    bn_2 = BatchNormalization()(act_1)
    conv_2 = Conv2D(32, (3, 3), padding='same')(bn_2)
    act_2 = Activation('relu')(conv_2)
    merged_2 = concatenate([act_2, merged_1], axis=-1)

    bn_1 = BatchNormalization()(merged_2)
    conv_1 = Conv2D(128, (1, 1), padding='same')(bn_1)
    act_1 = Activation('relu')(conv_1)
    bn_2 = BatchNormalization()(act_1)
    conv_2 = Conv2D(32, (3, 3), padding='same')(bn_2)
    act_2 = Activation('relu')(conv_2)
    merged_3 = concatenate([act_2, merged_2], axis=-1)

    bn_1 = BatchNormalization()(merged_3)
    conv_1 = Conv2D(128, (1, 1), padding='same')(bn_1)
    act_1 = Activation('relu')(conv_1)
    bn_2 = BatchNormalization()(act_1)
    conv_2 = Conv2D(32, (3, 3), padding='same')(bn_2)
    act_2 = Activation('relu')(conv_2)
    merged_4 = concatenate([act_2, merged_3], axis=-1)

    ###### Transition layer 1 #####

    conv_1 = Conv2D(128, (1, 1), padding='same')(merged_4)
    act_1 = Activation('relu')(conv_1)
    avg_p_1 = AveragePooling2D(strides=2)(act_1)


    ##### DENSE BLOCK 2 #####

    bn_1 = BatchNormalization()(avg_p_1)
    conv_1 = Conv2D(128, (1, 1), padding='same')(bn_1)
    act_1 = Activation('relu')(conv_1)
    bn_2 = BatchNormalization()(act_1)
    conv_2 = Conv2D(32, (3, 3), padding='same')(bn_2)
    act_2 = Activation('relu')(conv_2)
    merged_1 = concatenate([act_2, avg_p_1], axis=-1)

    bn_1 = BatchNormalization()(merged_1)
    conv_1 = Conv2D(128, (1, 1), padding='same')(bn_1)
    act_1 = Activation('relu')(conv_1)
    bn_2 = BatchNormalization()(act_1)
    conv_2 = Conv2D(32, (3, 3), padding='same')(bn_2)
    act_2 = Activation('relu')(conv_2)
    merged_2 = concatenate([act_2, merged_1], axis=-1)

    bn_1 = BatchNormalization()(merged_2)
    conv_1 = Conv2D(128, (1, 1), padding='same')(bn_1)
    act_1 = Activation('relu')(conv_1)
    bn_2 = BatchNormalization()(act_1)
    conv_2 = Conv2D(32, (3, 3), padding='same')(bn_2)
    act_2 = Activation('relu')(conv_2)
    merged_3 = concatenate([act_2, merged_2], axis=-1)

    bn_1 = BatchNormalization()(merged_3)
    conv_1 = Conv2D(128, (1, 1), padding='same')(bn_1)
    act_1 = Activation('relu')(conv_1)
    bn_2 = BatchNormalization()(act_1)
    conv_2 = Conv2D(32, (3, 3), padding='same')(bn_2)
    act_2 = Activation('relu')(conv_2)
    merged_4 = concatenate([act_2, merged_3], axis=-1)

    ###### Transition layer 2 #####

    conv_1 = Conv2D(128, (1, 1), padding='same')(merged_4)
    act_1 = Activation('relu')(conv_1)
    avg_p_1 = AveragePooling2D(strides=2)(act_1)


    ##### DENSE BLOCK 3 #####

    bn_1 = BatchNormalization()(avg_p_1)
    conv_1 = Conv2D(128, (1, 1), padding='same')(bn_1)
    act_1 = Activation('relu')(conv_1)
    bn_2 = BatchNormalization()(act_1)
    conv_2 = Conv2D(32, (3, 3), padding='same')(bn_2)
    act_2 = Activation('relu')(conv_2)
    merged_1 =concatenate([act_2, avg_p_1], axis=-1)

    bn_1 = BatchNormalization()(merged_1)
    conv_1 = Conv2D(128, (1, 1), padding='same')(bn_1)
    act_1 = Activation('relu')(conv_1)
    bn_2 = BatchNormalization()(act_1)
    conv_2 = Conv2D(32, (3, 3), padding='same')(bn_2)
    act_2 = Activation('relu')(conv_2)
    merged_2 =concatenate([act_2, merged_1], axis=-1)

    bn_1 = BatchNormalization()(merged_2)
    conv_1 = Conv2D(128, (1, 1), padding='same')(bn_1)
    act_1 = Activation('relu')(conv_1)
    bn_2 = BatchNormalization()(act_1)
    conv_2 = Conv2D(32, (3, 3), padding='same')(bn_2)
    act_2 = Activation('relu')(conv_2)
    merged_3 =concatenate([act_2, merged_2], axis=-1)

    bn_1 = BatchNormalization()(merged_3)
    conv_1 = Conv2D(128, (1, 1), padding='same')(bn_1)
    act_1 = Activation('relu')(conv_1)
    bn_2 = BatchNormalization()(act_1)
    conv_2 = Conv2D(32, (3, 3), padding='same')(bn_2)
    act_2 = Activation('relu')(conv_2)
    merged_4 =concatenate([act_2, merged_3], axis=-1)

    ###### Transition layer 3 #####

    conv_1 = Conv2D(128, (1, 1), padding='same')(merged_4)
    act_1 = Activation('relu')(conv_1)
    avg_p_1 = AveragePooling2D(strides=2)(act_1)

    ##### DENSE BLOCK 4 #####

    bn_1 = BatchNormalization()(avg_p_1)
    conv_1 = Conv2D(128, (1, 1), padding='same')(bn_1)
    act_1 = Activation('relu')(conv_1)
    bn_2 = BatchNormalization()(act_1)
    conv_2 = Conv2D(32, (3, 3), padding='same')(bn_2)
    act_2 = Activation('relu')(conv_2)
    merged_1 = concatenate([act_2, avg_p_1], axis=-1)

    bn_1 = BatchNormalization()(merged_1)
    conv_1 = Conv2D(128, (1, 1), padding='same')(bn_1)
    act_1 = Activation('relu')(conv_1)
    bn_2 = BatchNormalization()(act_1)
    conv_2 = Conv2D(32, (3, 3), padding='same')(bn_2)
    act_2 = Activation('relu')(conv_2)
    merged_2 = concatenate([act_2, merged_1], axis=-1)

    bn_1 = BatchNormalization()(merged_2)
    conv_1 = Conv2D(128, (1, 1), padding='same')(bn_1)
    act_1 = Activation('relu')(conv_1)
    bn_2 = BatchNormalization()(act_1)
    conv_2 = Conv2D(32, (3, 3), padding='same')(bn_2)
    act_2 = Activation('relu')(conv_2)
    merged_3 = concatenate([act_2, merged_2], axis=-1)

    bn_1 = BatchNormalization()(merged_3)
    conv_1 = Conv2D(128, (1, 1), padding='same')(bn_1)
    act_1 = Activation('relu')(conv_1)
    bn_2 = BatchNormalization()(act_1)
    conv_2 = Conv2D(32, (3, 3), padding='same')(bn_2)
    act_2 = Activation('relu')(conv_2)
    merged_4 = concatenate([act_2, merged_3], axis=-1)

    ###### Transition layer 4 #####

    conv_1 = Conv2D(128, (1, 1), padding='same')(merged_4)
    act_1 = Activation('relu')(conv_1)
    avg_p_1 = AveragePooling2D(strides=2)(act_1)

    ##### DENSE BLOCK 5 #####

    bn_1 = BatchNormalization()(avg_p_1)
    conv_1 = Conv2D(128, (1, 1), padding='same')(bn_1)
    act_1 = Activation('relu')(conv_1)
    bn_2 = BatchNormalization()(act_1)
    conv_2 = Conv2D(32, (3, 3), padding='same')(bn_2)
    act_2 = Activation('relu')(conv_2)
    merged_1 = concatenate([act_2, avg_p_1], axis=-1)

    bn_1 = BatchNormalization()(merged_1)
    conv_1 = Conv2D(128, (1, 1), padding='same')(bn_1)
    act_1 = Activation('relu')(conv_1)
    bn_2 = BatchNormalization()(act_1)
    conv_2 = Conv2D(32, (3, 3), padding='same')(bn_2)
    act_2 = Activation('relu')(conv_2)
    merged_2 = concatenate([act_2, merged_1], axis=-1)

    bn_1 = BatchNormalization()(merged_2)
    conv_1 = Conv2D(128, (1, 1), padding='same')(bn_1)
    act_1 = Activation('relu')(conv_1)
    bn_2 = BatchNormalization()(act_1)
    conv_2 = Conv2D(32, (3, 3), padding='same')(bn_2)
    act_2 = Activation('relu')(conv_2)
    merged_3 = concatenate([act_2, merged_2], axis=-1)

    bn_1 = BatchNormalization()(merged_3)
    conv_1 = Conv2D(128, (1, 1), padding='same')(bn_1)
    act_1 = Activation('relu')(conv_1)
    bn_2 = BatchNormalization()(act_1)
    conv_2 = Conv2D(32, (3, 3), padding='same')(bn_2)
    act_2 = Activation('relu')(conv_2)
    merged_4 = concatenate([act_2, merged_3], axis=-1)

    ###### Transition layer 5 #####

    conv_1 = Conv2D(128, (1, 1), padding='same')(merged_4)
    act_1 = Activation('relu')(conv_1)
    avg_p_1 = AveragePooling2D(strides=2)(act_1)

    ##### DENSE BLOCK 6 #####

    bn_1 = BatchNormalization()(avg_p_1)
    conv_1 = Conv2D(128, (1, 1), padding='same')(bn_1)
    act_1 = Activation('relu')(conv_1)
    bn_2 = BatchNormalization()(act_1)
    conv_2 = Conv2D(32, (3, 3), padding='same')(bn_2)
    act_2 = Activation('relu')(conv_2)
    merged_1 = concatenate([act_2, avg_p_1], axis=-1)

    bn_1 = BatchNormalization()(merged_1)
    conv_1 = Conv2D(128, (1, 1), padding='same')(bn_1)
    act_1 = Activation('relu')(conv_1)
    bn_2 = BatchNormalization()(act_1)
    conv_2 = Conv2D(32, (3, 3), padding='same')(bn_2)
    act_2 = Activation('relu')(conv_2)
    merged_2 = concatenate([act_2, merged_1], axis=-1)

    bn_1 = BatchNormalization()(merged_2)
    conv_1 = Conv2D(128, (1, 1), padding='same')(bn_1)
    act_1 = Activation('relu')(conv_1)
    bn_2 = BatchNormalization()(act_1)
    conv_2 = Conv2D(32, (3, 3), padding='same')(bn_2)
    act_2 = Activation('relu')(conv_2)
    merged_3 = concatenate([act_2, merged_2], axis=-1)

    bn_1 = BatchNormalization()(merged_3)
    conv_1 = Conv2D(128, (1, 1), padding='same')(bn_1)
    act_1 = Activation('relu')(conv_1)
    bn_2 = BatchNormalization()(act_1)
    conv_2 = Conv2D(32, (3, 3), padding='same')(bn_2)
    act_2 = Activation('relu')(conv_2)
    merged_4 = concatenate([act_2, merged_3], axis=-1)


    ## Dense Layer with GlobalAveragePooling
    global_avg_p = GlobalAveragePooling2D()(merged_4)
    denselayer = Dense(3000, activation='elu', kernel_regularizer=l2(0.001))(global_avg_p)
    # output = Activation('softmax')(denselayer)


    model = Model(inputs=[inputs], outputs=[denselayer])

    return model


if __name__ == '__main__':

    pass
