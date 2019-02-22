

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
def build_new_model(input_shape):

    num_classes = 1383

    model = keras.applications.DenseNet169(input_shape=input_shape, include_top=False)
    model.trainable = False

    x1_1 = GlobalAveragePooling2D()(model.layers[-1].output)
    x1_2 = GlobalAveragePooling2D()(model.layers[-9].output)
    x1_3 = GlobalAveragePooling2D()(model.layers[-23].output)

    x2_1 = Dense(512, activation='elu')(x1_1)
    x2_2 = Dense(512, activation='elu')(x1_2)
    x2_3 = Dense(512, activation='elu')(x1_3)

    con = concatenate([x2_1, x2_2, x2_3], axis=-1)
    con = Dropout(.5)(con)
    con2 = Dense(num_classes, activation='elu')(con)

    con3 = Lambda(Cosine_theta)(con2)

    model_new = Model(inputs=model.input, outputs=con3)
    model_new.summary()

    return model_new


def build_DenseNet169_pretrained(input_shape):

    model = keras.applications.DenseNet169(input_shape=input_shape, include_top=False)
    # model = keras.applications.DenseNet201(input_shape=input_shape, include_top=False)
    model.trainable = False
    x1 = GlobalAveragePooling2D()(model.output)
    x2 = Lambda(Cosine_theta)(x1)

    model_new = Model(inputs=model.input, outputs=x2)
    model_new.summary()

    return model_new

import math

def Cosine_theta(x):

    embedding_dim = 1383
    # embedding_dim = 1920
    num_classes = 1383

    features = x

    var_weights = tf.Variable(constant_xavier_initializer([num_classes, embedding_dim]), name='weights')
    normed_weights = tf.nn.l2_normalize(var_weights, 1, 1e-10, name='weights_norm')
    normed_features = tf.nn.l2_normalize(features, 1, 1e-10, name='features_norm')

    cosine = tf.matmul(normed_features, normed_weights, transpose_a=False, transpose_b=True)

    return cosine

def constant_xavier_initializer(shape, dtype=tf.float32, uniform=True):
    """Initializer function."""
    if not dtype.is_floating:
        raise TypeError('Cannot create initializer for non-floating point type.')

    if shape:
        fan_in = float(shape[-2]) if len(shape) > 1 else float(shape[-1])
        fan_out = float(shape[-1])
    else:
        fan_in = 1.0
        fan_out = 1.0
    for dim in shape[:-2]:
        fan_in *= float(dim)
        fan_out *= float(dim)

    # Average number of inputs and output connections.
    n = (fan_in + fan_out) / 2.0
    if uniform:
        # To get stddev = math.sqrt(factor / n) need to adjust for uniform.
        limit = math.sqrt(3.0 * 1.0 / n)
        return tf.random_uniform(shape, -limit, limit, dtype, seed=None)
    else:
        # To get stddev = math.sqrt(factor / n) need to adjust for truncated.
        trunc_stddev = math.sqrt(1.3 * 1.0 / n)
        return tf.truncated_normal(shape, 0.0, trunc_stddev, dtype, seed=None)


if __name__ == '__main__':

    pass
