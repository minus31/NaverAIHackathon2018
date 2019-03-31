from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import argparse
import time
import numpy as np
import math
import keras
import tensorflow as tf


from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, Lambda
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import preprocess_input

from build_model import build_DenseNet169_pretrained
from build_model import build_new_model
# Score
# checkpoint - score
#


def bind_model(model):
    def save(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        model.save_weights(os.path.join(dir_name, 'model'))
        print('model saved!')

    def load(file_path):
        model.load_weights(file_path)
        print('model loaded!')

    def infer(queries, _):
        test_path = DATASET_PATH + '/test/test_data'

        db = [os.path.join(test_path, 'reference', path) for path in os.listdir(os.path.join(test_path, 'reference'))]

        queries = [v.split('/')[-1].split('.')[0] for v in queries]
        db = [v.split('/')[-1].split('.')[0] for v in db]
        queries.sort()
        db.sort()

        queries, query_vecs, references, reference_vecs = get_feature(model, queries, db)

        # l2 normalization
        query_vecs = l2_normalize(query_vecs)
        reference_vecs = l2_normalize(reference_vecs)


        # Calculate cosine similarity
        sim_matrix = np.dot(query_vecs, reference_vecs.T)


        # neg_dist_mat = np.asarray([[l2_distance(p, r) for r in reference_vecs] for p in query_vecs])


        indices = np.argsort(sim_matrix, axis=1)
        # indices = np.argsort(neg_dist_mat, axis=1)
        indices = np.flip(indices, axis=1)

        retrieval_results = {}

        for (i, query) in enumerate(queries):
            ranked_list = [references[k] for k in indices[i]]
            ranked_list = ranked_list[:1000]

            retrieval_results[query] = ranked_list
        print('done')

        return list(zip(range(len(retrieval_results)), retrieval_results.items()))

def l2_distance(p, q):
    return np.linalg.norm(p - q)

def l2_normalize(v):

    norm = np.linalg.norm(v, axis=1, keepdims=True)

    return np.divide(v, norm, where=norm!=0)


# data preprocess
def get_feature(model, queries, db):
    img_size = (224, 224)
    test_path = DATASET_PATH + '/test/test_data'


    intermediate_layer_model = Model(inputs=model.input, outputs=model.layers[-2].output)
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input ,dtype='float32')
    query_generator = test_datagen.flow_from_directory(
        directory=test_path,
        target_size=(224, 224),
        classes=['query'],
        color_mode="rgb",
        batch_size=32,
        class_mode=None,
        shuffle=False
    )
    query_vecs = intermediate_layer_model.predict_generator(query_generator, steps=len(query_generator), verbose=1)

    reference_generator = test_datagen.flow_from_directory(
        directory=test_path,
        target_size=(224, 224),
        classes=['reference'],
        color_mode="rgb",
        batch_size=32,
        class_mode=None,
        shuffle=False
    )
    reference_vecs = intermediate_layer_model.predict_generator(reference_generator, steps=len(reference_generator),
                                                                verbose=1)

    return queries, query_vecs, db, reference_vecs

def ArcLoss2(labels, features):

    N = tf.shape(labels)[0]
    s = 64.
    m1 = 1.
    m2 = 0.5
    m3 = 0.

    target_cos = tf.reduce_sum(tf.cast(labels, tf.float32) * features, axis=-1)
    target_cos = tf.cos(tf.math.acos(target_cos) * m1 + m2) - m3
    target_cos = tf.exp(s * target_cos)

    others = tf.multiply(tf.subtract(tf.cast(labels, tf.float32), 1.0), features)
    others = tf.exp(s * others)
    others = tf.reduce_sum(others, axis=-1)

    log_ = tf.log(tf.divide(target_cos, tf.add(target_cos, others)))

    output = -1. * tf.divide(tf.reduce_sum(log_), tf.cast(N, tf.float32))

    return output

def FocalLoss(labels, features, num_classes=1383, gamma=1.0, scope=None):

    one_hot = labels
    prob = tf.nn.softmax(features)

    return tf.reduce_mean(tf.reduce_sum(one_hot * (0. - tf.pow(1. - prob, gamma) * tf.nn.log_softmax(features)), axis=-1), name='focal_loss')


if __name__ == '__main__':
    args = argparse.ArgumentParser()

    # hyperparameters
    args.add_argument('--epoch', type=int, default=100)
    args.add_argument('--batch_size', type=int, default=64)
    args.add_argument('--num_classes', type=int, default=1383)

    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train', help='submit일때 해당값이 test로 설정됩니다.')
    args.add_argument('--iteration', type=str, default='0',
                      help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')
    config = args.parse_args()

    # training parameters
    nb_epoch = config.epoch
    batch_size = config.batch_size
    num_classes = config.num_classes
    input_shape = (224, 224, 3)  # input image shape

    model = build_new_model(input_shape)

    if config.mode == 'train':

        """ Initiate RMSprop optimizer """
        opt = keras.optimizers.rmsprop(lr=0.00045, decay=1e-6)
        model.compile(loss=FocalLoss,
                      optimizer=opt)
        print('dataset path', DATASET_PATH)
        train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
        zoom_range=0.2, vertical_flip=True,
        horizontal_flip=True, validation_split=0.1)

        train_generator = train_datagen.flow_from_directory(
            directory=DATASET_PATH + '/train/train_data',
            target_size=input_shape[:2],
            color_mode="rgb",
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=True,
            subset='training'
        )
        val_generator = train_datagen.flow_from_directory(
            directory=DATASET_PATH + '/train/train_data',
            target_size=input_shape[:2],
            color_mode="rgb",
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=True,
            subset='validation'
        )

        """ Callback """
        monitor = 'loss'
        reduce_lr = ReduceLROnPlateau(monitor=monitor, patience=3)

        """ Training loop """
        STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
        STEP_SIZE_VAL = val_generator.n // val_generator.batch_size
        t0 = time.time()
        for epoch in range(nb_epoch):
            t1 = time.time()
            res = model.fit_generator(generator=train_generator,
                                      steps_per_epoch=STEP_SIZE_TRAIN,
                                      initial_epoch=epoch,
                                      validation_data = val_generator,
                                      validation_steps = STEP_SIZE_VAL,
                                      epochs=epoch + 1,
                                      callbacks=[reduce_lr],
                                      verbose=1,
                                      shuffle=True)
            t2 = time.time()
            print(res.history)
            print('Training time for one epoch : %.1f' % ((t2 - t1)))
            val_loss = res.history['val_loss'][0]
        print('Total training time : %.1f' % (time.time() - t0))
