from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import argparse
import time
import nsml
import numpy as np
import math
from nsml import DATASET_PATH
import keras
import tensorflow as tf


from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, Lambda
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import preprocess_input

from build_model import build_DenseNet169_pretrained

# Score
# checkpoint - score
#      0     - 0.6787675138977849


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
        indices = np.argsort(sim_matrix, axis=1)
        indices = np.flip(indices, axis=1)

        retrieval_results = {}

        for (i, query) in enumerate(queries):
            ranked_list = [references[k] for k in indices[i]]
            ranked_list = ranked_list[:1000]

            retrieval_results[query] = ranked_list
        print('done')

        return list(zip(range(len(retrieval_results)), retrieval_results.items()))

    # DONOTCHANGE: They are reserved for nsml
    nsml.bind(save=save, load=load, infer=infer)


def l2_normalize(v):

    norm = np.linalg.norm(v, axis=1, keepdims=True)

    return np.divide(v, norm, where=norm!=0)


# data preprocess
def get_feature(model, queries, db):
    img_size = (224, 224)
    test_path = DATASET_PATH + '/test/test_data'

    intermediate_layer_model = Model(inputs=model.input, outputs=model.layers[-1].output)
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


def ArcFaceLoss(labels, features):

    scale=64.
    margin=0.8
    embedding_dim = 1664
    num_classes=1383
    easy_margin=True
    scope=None

    cosine = features
    cos_m = math.cos(margin)
    sin_m = math.sin(margin)
    mm = math.sin(math.pi - margin) * margin
    threshold = math.cos(math.pi - margin)

    one_hot_mask = tf.cast(labels, tf.float32)
    labels = tf.argmax(labels, axis=-1)

    cosine_theta_2 = tf.pow(cosine, 2., name='cosine_theta_2') # 가중치와 피쳐 상의 각을 제곱
    sine_theta = tf.pow(1. - cosine_theta_2, .5, name='sine_theta') #

    cosine_theta_m = scale * (cos_m * cosine - sin_m * sine_theta) * one_hot_mask

    clip_mask = tf.to_float(cosine >= threshold) * scale * mm * one_hot_mask

    cosine = scale * cosine * (1. - one_hot_mask) + tf.where(clip_mask > 0., cosine_theta_m, clip_mask)

    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(labels, tf.int64),
                                                                        logits=cosine), name='arc_loss')

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

    model = build_DenseNet169_pretrained(input_shape)

    bind_model(model)

    if config.pause:
        nsml.paused(scope=locals())

    bTrainmode = False
    if config.mode == 'train':
        bTrainmode = True

        """ Initiate RMSprop optimizer """
        opt = keras.optimizers.rmsprop(lr=0.00045, decay=1e-6)
        model.compile(loss=ArcFaceLoss,
                      optimizer=opt)
        nsml.save('bt')
        print('dataset path', DATASET_PATH)

        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

        train_generator = train_datagen.flow_from_directory(
            directory=DATASET_PATH + '/train/train_data',
            target_size=input_shape[:2],
            color_mode="rgb",
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=True
        )

        """ Callback """
        monitor = 'loss'
        reduce_lr = ReduceLROnPlateau(monitor=monitor, patience=3)

        """ Training loop """
        STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
        t0 = time.time()
        for epoch in range(nb_epoch):
            t1 = time.time()
            res = model.fit_generator(generator=train_generator,
                                      steps_per_epoch=STEP_SIZE_TRAIN,
                                      initial_epoch=epoch,
                                      epochs=epoch + 1,
                                      callbacks=[reduce_lr],
                                      verbose=1,
                                      shuffle=True)
            t2 = time.time()
            print(res.history)
            print('Training time for one epoch : %.1f' % ((t2 - t1)))
            train_loss = res.history['loss'][0]
            nsml.report(summary=True, epoch=epoch, epoch_total=nb_epoch, loss=train_loss)
            nsml.save(epoch)
        print('Total training time : %.1f' % (time.time() - t0))
