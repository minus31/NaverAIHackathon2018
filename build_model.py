

# -*- coding: utf_8 -*-

import os
import cv2
import pickle
import tensorflow as tf
import math
import numpy as np
import keras
from keras.models import Sequential
from keras.callbacks import ReduceLROnPlateau
from keras import backend as K
from keras.engine.input_layer import Input
from keras.models import Model
from keras import regularizers
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers import BatchNormalization, Lambda, AveragePooling2D
from keras.layers import GlobalAveragePooling2D, Activation, concatenate
from keras.regularizers import l2
from keras.utils import multi_gpu_model




"""
Model Architecture

- build_model()
"""

def build_DenseNet169_pretrained(input_shape):

    # model = keras.applications.DenseNet169(input_shape=input_shape, include_top=False)
    model = keras.applications.DenseNet201(input_shape=input_shape, include_top=False)
    model.trainable = False
    x1 = GlobalAveragePooling2D()(model.output)
    x2 = Lambda(Cosine_theta)(x1)

    model_new = Model(inputs=model.input, outputs=x2)
    model_new.summary()

    return model_new

import math

def Cosine_theta(x):

    # embedding_dim = 1664
    embedding_dim = 1920
    num_classes = 1383

    features = x

    # var_weights = tf.Variable(constant_xavier_initializer([num_classes, embedding_dim]), name='weights')
    with tf.variable_scope("Cosine_theta", reuse=tf.AUTO_REUSE):

        var_weights = tf.get_variable(name='weight4cos',shape=(num_classes, embedding_dim),
                                        dtype=tf.float32,
                                        initializer=tf.random_uniform_initializer)

        normed_weights = tf.nn.l2_normalize(var_weights, 1, 1e-10, name='weights_norm')
        normed_features = tf.nn.l2_normalize(features, 1, 1e-10, name='features_norm')

        cosine = tf.matmul(normed_features, normed_weights, transpose_a=False, transpose_b=True)

    return cosine


if __name__ == '__main__':

    pass
