# -*- coding: utf_8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import argparse
import pickle
import nsml
from nsml import DATASET_PATH
import numpy as np
import keras
from keras.models import Sequential
from keras.callbacks import ReduceLROnPlateau
from keras import backend as K
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.engine.input_layer import Input
from keras.models import Model
from keras import regularizers
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Lambda, AveragePooling2D, GlobalAveragePooling2D, Activation, concatenate
from keras.regularizers import l2

from data_loader import train_data_loader


# https://github.com/maciejkula/triplet_recommendations_keras/blob/master/triplet_keras.ipynb

def bind_model(base_model):
    def save(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        base_model.save_weights(os.path.join(dir_name, 'model'))
        print('base model saved!')

    def load(file_path):
        base_model.load_weights(file_path)
        # model.load_weights("./models/GodGam_ir_ph1_v2_1/699/model/model")
        print('base model loaded!')

    def infer(queries, db):

        # Query 개수: 195
        # Reference(DB) 개수: 1,127
        # Total (query + reference): 1,1282

        queries, query_img, references, reference_img = preprocess(queries, db)

        print('test data load queries {} query_img {} references {} reference_img {}'.
              format(len(queries), len(query_img), len(references), len(reference_img)))

        queries = np.asarray(queries)
        query_img = np.asarray(query_img)
        references = np.asarray(references)
        reference_img = np.asarray(reference_img)

        query_img = query_img.astype('float32')
        query_img = preprocess_input(query_img)

        reference_img = reference_img.astype('float32')
        reference_img = preprocess_input(reference_img)

        get_feature_layer = K.function([base_model.layers[0].input] + [K.learning_phase()], [base_model.layers[-2].output])

        print('Triplet inference start')
        # inference
        query_vecs = get_feature_layer([query_img, 0])[0]

        # caching db output, db inference

        db_output = './db_infer.pkl'
        if os.path.exists(db_output):
            with open(db_output, 'rb') as f:
                reference_vecs = pickle.load(f)
        else:
            reference_vecs = get_feature_layer([reference_img, 0])[0]
            with open(db_output, 'wb') as f:
                pickle.dump(reference_vecs, f)

        retrieval_results = {}

        for (i, query) in enumerate(queries):
            query = query.split('/')[-1].split('.')[0]
            l2_dict = {}
            for (j, ref) in enumerate(references):
                ref = ref.split('/')[-1].split('.')[0]
                l2_dict[ref] = l2_distance(query_vecs[i], reference_vecs[j])

            sorted_l2 = sorted(l2_dict.items(), key=lambda x: x[1])
            sorted_l2 = [x[0] for x in sorted_l2]

            retrieval_results[query] = sorted_l2
        print('done')

        print(list(zip(range(len(retrieval_results)), retrieval_results.items())))
        return list(zip(range(len(retrieval_results)), retrieval_results.items()))

    # DONOTCHANGE: They are reserved for nsml
    nsml.bind(save=save, load=load, infer=infer)

def l2_distance(q, r):

    return np.linalg.norm(q - r)

def l2_distanceK(q, r):

    return K.sqrt(K.sum((q - r)**2))

def triplet_loss(X, margin=0.5):
    [a, p, n] = X

    return K.max([0, (l2_distanceK(a, p) - l2_distanceK(a, n) + margin)])
# def triplet_loss(X):
#
#     user_latent, positive_item_latent, negative_item_latent = X
#     # BPR loss
#     loss = 1.0 - K.sigmoid(
#         K.sum(user_latent * positive_item_latent, axis=-1, keepdims=True) -
#         K.sum(user_latent * negative_item_latent, axis=-1, keepdims=True))
#
#     return loss


def identity_loss(y_true, y_pred):

    return K.mean(y_pred - 0 * y_true)


def get_related_img(x_train, labels, model):

    """selector for triplet loss """

    idx = np.arange(len(x_train))
    input_shape = x_train.shape[1:]
    model_ = model
    get_feature_layer = K.function([model_.layers[0].input] + [K.learning_phase()], [model_.layers[-2].output])

    for i in range(7):
        if i == 0:
            feature_vec = get_feature_layer([x_train[:1000], 0])[0]
        elif i == 6:
            f = get_feature_layer([x_train[i*1000:], 0])[0]
            feature_vec = np.concatenate((feature_vec,f), axis=0)
            del f
            print("get_feature_vecter !!")
        else :
            f = get_feature_layer([x_train[i*1000:(i+1)*1000], 0])[0]
            feature_vec = np.concatenate((feature_vec,f), axis=0)
        print(i)
    # feature_vec =

    positive_train = []
    negative_train = []

    for i in idx:
        print("progress {}/{}".format(i, x_train.shape[0]))
        candidates_id = idx[labels == labels[i]]
        candidates_vec = feature_vec[candidates_id]
        dist_ls = []
        for vec in candidates_vec:
            dist = l2_distance(feature_vec[i], vec)
            dist_ls.append(dist)

        max_ix = np.argmax(dist_ls)
        max_idx = candidates_id[max_ix]
        print("max_index", max_idx)
        positive_train.append(x_train[max_idx])

        candidates_id = np.random.choice(idx[labels != labels[i]], 100)
        candidates_vec = feature_vec[candidates_id]
        dist_ls = []

        for vec in candidates_vec:
            dist = l2_distance(feature_vec[i], vec)
            dist_ls.append(dist)

        min_ix = np.argmin(dist_ls)
        min_idx = candidates_id[min_ix]
        print("max_index", min_idx)
        negative_train.append(x_train[min_idx])

    print(len(negative_train), len(positive_train))

    negative_train = np.asarray(negative_train)
    positive_train = np.asarray(positive_train)
    return positive_train, negative_train

# data preprocess
def preprocess(queries, db):
    query_img = []
    reference_img = []
    img_size = (224, 224)

    for img_path in queries:
        img = cv2.imread(img_path, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size)
        query_img.append(img)

    for img_path in db:
        img = cv2.imread(img_path, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size)
        reference_img.append(img)

    return queries, query_img, db, reference_img

if __name__ == '__main__':
    args = argparse.ArgumentParser()

    # hyperparameters
    args.add_argument('--epochs', type=int, default=300)
    args.add_argument('--batch_size', type=int, default=128)

    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train', help='submit일때 해당값이 test로 설정됩니다.')
    args.add_argument('--iteration', type=str, default='0', help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')
    config = args.parse_args()


    # training parameters
    nb_epoch = config.epochs
    batch_size = config.batch_size
    num_classes = 1000
    input_shape = (224, 224, 3)  # input image shape



    # Model to Train
    base_model = ResNet50(input_shape=input_shape, include_top=False, weights='imagenet')
    x = GlobalAveragePooling2D()(base_model.output)
    base_model = Model(inputs=base_model.input, outputs=x)

    # input_tensor = Input(shape=(3, 224, 224, 3))

    anchor_input = Input(shape=input_shape)
    positive_input = Input(shape=input_shape)
    negative_input = Input(shape=input_shape)

    anchor_vec = base_model(anchor_input)
    positive_vec = base_model(positive_input)
    negative_vec = base_model(negative_input)
    loss = Lambda(triplet_loss)([anchor_vec, positive_vec, negative_vec])
    model = Model(inputs=[anchor_input, positive_input, negative_input], outputs=loss)

    model.summary()

    bind_model(base_model)

    if config.pause:
        nsml.paused(scope=locals())

    bTrainmode = False
    if config.mode == 'train':
        bTrainmode = True

        """ Initiate RMSprop optimizer """
        op = keras.optimizers.rmsprop(lr=0.00045, decay=1e-6)
        model.compile(loss=identity_loss,
                      optimizer=op)


        """ Load data """
        print('dataset path', DATASET_PATH)
        output_path = ['./img_list.pkl', './label_list.pkl']
        train_dataset_path = DATASET_PATH + '/train/train_data'

        if nsml.IS_ON_NSML:
            # Caching file
            nsml.cache(train_data_loader, data_path=train_dataset_path, img_size=input_shape[:2],
                       output_path=output_path)
        else:
            # local에서 실험할경우 dataset의 local-path 를 입력해주세요.
            train_data_loader(train_dataset_path, input_shape[:2], output_path=output_path)

        with open(output_path[0], 'rb') as img_f:
            img_list = pickle.load(img_f)
        with open(output_path[1], 'rb') as label_f:
            label_list = pickle.load(label_f)

        # Preprocessing
        x_train = np.asarray(img_list)
        labels = np.asarray(label_list)
        # y_train = keras.utils.to_categorical(labels, num_classes=num_classes)
        x_train = x_train.astype('float32')
        x_train = preprocess_input(x_train)

        # get_related_img
        positive_train, negative_train = get_related_img(x_train, labels, base_model)

        # x_train_tri = [x_train, positive_train, negative_train]

        """ Callback """
        monitor = 'loss'
        reduce_lr = ReduceLROnPlateau(monitor=monitor, patience=3)

        """ Training loop """
        for epoch in range(nb_epoch):
            res = model.fit([x_train, positive_train, negative_train], labels,
                            batch_size=batch_size,
                            initial_epoch=epoch,
                            epochs=epoch + 1,
                            callbacks=[reduce_lr],
                            verbose=1,
                            shuffle=True)
            print(res.history)
            train_loss = res.history['loss'][0]
            nsml.report(summary=True, epoch=epoch, epoch_total=nb_epoch, loss=train_loss)
            if epoch % 50 == 0:
                positive_train, negative_train = get_related_img(x_train, labels, base_model)
            if epoch > 100 :
                nsml.save(epoch)
        print(model.layers)
