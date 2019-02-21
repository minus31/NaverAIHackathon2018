

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

    model = keras.applications.DenseNet169(input_shape=input_shape, include_top=False)
    x1 = GlobalAveragePooling2D()(model.output)
    x2 = Lambda(ArcFace)(x1)

    model_new = Model(inputs=model.input, outputs=x2)
    model_new.summary()

    return model_new

def build_resnet_pretrained(input_shape):

    from keras.applications.resnet50 import ResNet50

    model = ResNet50(input_shape=input_shape, include_top=False)
    x1 = GlobalAveragePooling2D()(model.output)
    x2 = Lambda(ArcFace)(x1)

    model_new = Model(inputs=model.input, outputs=x2)
    model_new.summary()

    return model_new

import math

def ArcFace(x):

    embedding_dim = 1664
    num_classes = 1383
    margin = 0.5
    features = x

    cos_m = math.cos(margin)
    sin_m = math.sin(margin)
    mm = math.sin(math.pi - margin) * margin
    threshold = math.cos(math.pi - margin)

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
