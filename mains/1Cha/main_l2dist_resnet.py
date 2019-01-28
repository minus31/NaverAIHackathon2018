# -*- coding: utf_8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import cv2
import argparse
import pickle

import nsml
import numpy as np

from nsml import DATASET_PATH
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ReduceLROnPlateau
from keras import backend as K
from data_loader import train_data_loader
from keras.applications.resnet50 import ResNet50, preprocess_input


def bind_model(model):
    def save(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        model.save_weights(os.path.join(dir_name, 'model'))
        print('model saved!')

    def load(file_path):
        model.load_weights(file_path)
        # model.load_weights("./models/GodGam_ir_ph1_v2_1/699/model/model")
        print('model loaded!')

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
        # query_img /= 255
        reference_img = reference_img.astype('float32')
        reference_img = preprocess_input(reference_img)
        # reference_img /= 255

        # get_product_class = K.function([model.layers[0].input] + [K.learning_phase()], [model.layers[-1].output])
        get_feature_layer = K.function([model.layers[0].input] + [K.learning_phase()], [model.layers[-2].output])

        print('Mine inference start')

        # inference
        # query_class = get_product_class([query_img, 0])[0]
        # query_class = np.argmax(query_class, axis=-1)
        # print(query_class)
        query_vecs = get_feature_layer([query_img, 0])[0]

        # caching db output, db inference

        # reference_class = get_product_class([reference_img, 0])[0]
        # reference_class = np.argmax(reference_class, axis=-1)

        db_output = './db_infer.pkl'
        if os.path.exists(db_output):
            with open(db_output, 'rb') as f:
                # reference_class = pickle.load(f)
                reference_vecs = pickle.load(f)
        else:
            reference_vecs = get_feature_layer([reference_img, 0])[0]
            with open(db_output, 'wb') as f:
                # pickle.dump(reference_class, f)
                pickle.dump(reference_vecs, f)
        # print(reference_class)
        # l2 normalization
        # query_vecs = l2_normalize(query_vecs)
        # reference_vecs = l2_normalize(reference_vecs)
        #
        # # Calculate cosine similarity
        # sim_matrix = np.dot(query_vecs, reference_vecs.T)

        # query_feature = [[0 if z < Z.mean() else 1 for z in Z] for Z in query_vecs]
        # reference_feature = [[0 if z < Z.mean() else 1 for z in Z] for Z in reference_vecs]

        # def hamming2(s1, s2):
        #     """Calculate the Hamming distance between two bit strings"""
        #     assert len(s1) == len(s2)
        #     return sum(c1 != c2 for c1, c2 in zip(s1, s2))

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

def l2_normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


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

from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.engine.input_layer import Input
from keras.models import Model
from keras import regularizers
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Concatenate, AveragePooling2D, GlobalAveragePooling2D, Activation, concatenate
from keras.regularizers import l2

if __name__ == '__main__':
    args = argparse.ArgumentParser()

    # hyperparameters
    args.add_argument('--epochs', type=int, default=1000)
    args.add_argument('--batch_size', type=int, default=256)

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

    # ```Model```

    model = ResNet50(input_shape=input_shape)
    # model = densenet(input_shape,num_classes)
    model.summary()

    bind_model(model)
    # model.load_weights("./models/GodGam_ir_ph1_v2_1/699/model/model")

    nsml.save("ResNet")
    print("ResNet")

    if config.pause:
        nsml.paused(scope=locals())

    bTrainmode = False
    # if config.mode == 'train':
        # bTrainmode = True
        #
        # """ Initiate RMSprop optimizer """
        # opt = keras.optimizers.rmsprop(lr=0.00045, decay=1e-6)
        # model.compile(loss='categorical_crossentropy',
        #               optimizer=opt,
        #               metrics=['accuracy'])
        #
        # """ Load data """
        # print('dataset path', DATASET_PATH)
        # output_path = ['./img_list.pkl', './label_list.pkl']
        # train_dataset_path = DATASET_PATH + '/train/train_data'
        #
        # if nsml.IS_ON_NSML:
        #     # Caching file
        #     nsml.cache(train_data_loader, data_path=train_dataset_path, img_size=input_shape[:2],
        #                output_path=output_path)
        # else:
        #     # local에서 실험할경우 dataset의 local-path 를 입력해주세요.
        #     train_data_loader(train_dataset_path, input_shape[:2], output_path=output_path)
        #
        # with open(output_path[0], 'rb') as img_f:
        #     img_list = pickle.load(img_f)
        # with open(output_path[1], 'rb') as label_f:
        #     label_list = pickle.load(label_f)
        #
        # x_train = np.asarray(img_list)
        # labels = np.asarray(label_list)
        # y_train = keras.utils.to_categorical(labels, num_classes=num_classes)
        # x_train = x_train.astype('float32')
        # x_train /= 255
        # print(len(labels), 'train samples')
        #
        # """ Callback """
        # monitor = 'acc'
        # reduce_lr = ReduceLROnPlateau(monitor=monitor, patience=3)
        #
        # """ Training loop """
        # for epoch in range(nb_epoch):
        #     res = model.fit(x_train, y_train,
        #                     batch_size=batch_size,
        #                     initial_epoch=epoch,
        #                     epochs=epoch + 1,
        #                     callbacks=[reduce_lr],
        #                     verbose=1,
        #                     shuffle=True)
        #     print(res.history)
        #     train_loss, train_acc = res.history['loss'][0], res.history['acc'][0]
        #     nsml.report(summary=True, epoch=epoch, epoch_total=nb_epoch, loss=train_loss, acc=train_acc)
        #     if epoch > 900 :
        #         nsml.save(epoch)
