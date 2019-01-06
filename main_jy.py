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
import torch

def bind_model(model):
    def save(dir_name):
        print(dir_name)
        os.makedirs(dir_name, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(dir_name, 'model.pth'))
        print('model saved!')

    def load(file_path):
        dir = file_path
        checkpoint = torch.load(dir, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint)        
        print('model loaded!')

    def infer(queries, db):
        # Query 개수: 195
        # Reference(DB) 개수: 1,127
        # Total (query + reference): 1,322
        
        queries, query_img, references, reference_img = preprocess(queries, db)

        print('test data load queries {} query_img {} references {} reference_img {}'.
              format(len(queries), len(query_img), len(references), len(reference_img)))

        # immediate model
        model_fc = model.features
        model_features = nn.Sequential(list(model.classifier.children())[0])

        transforms_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])            
            ])

        def hamming2(s1, s2):
            """Calculate the Hamming distance between two bit strings"""
            assert len(s1) == len(s2)
            return sum(c1 != c2 for c1, c2 in zip(s1, s2))

        # get feature of queries
        dict_query_features = {}
        for path, img in zip(queries, query_img):
            img_pre = transforms_test(img)
            output = model_fc(img_pre.cuda().unsqueeze(0))
            output = output.flatten()
            output = model_features(output)
            dict_query_features[path.split('/')[-1].split('.')[0]] = "".join((output > 0).cpu().detach().numpy().astype('str'))

        # get feature of references
        dict_reference_features = {}
        for path, img in zip(references, reference_img):
            img_pre = transforms_test(img)
            output = model_fc(img_pre.cuda().unsqueeze(0))
            output = output.flatten()
            output = model_features(output)
            dict_reference_features[path.split('/')[-1].split('.')[0]] = "".join((output > 0).cpu().detach().numpy().astype('str'))

        # make list
        li_result = []

        for query_key, query_features in dict_query_features.items():
            dict_hamming = {}
            for reference_key, reference_features in dict_reference_features.items():
                dict_hamming[reference_key] = hamming2(query_features, reference_features)
            
            sorted_hamming = sorted(dict_hamming.items(), key=lambda x: x[1])
            sorted_hamming = [x[0] for x in sorted_hamming]
            li_result.append((query_key, sorted_hamming))

        return list(zip(range(len(li_result)), li_result))        

    nsml.bind(save=save, load=load, infer=infer)

# input -> queries : path_query list, db : path_reference list
# return ->  queries(img, path), reference(img, path)

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

# model
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as utils


if __name__ == '__main__':
    args = argparse.ArgumentParser()

    # hyperparameters
    args.add_argument('--epochs', type=int, default=20)
    args.add_argument('--batch_size', type=int, default=64)

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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = models.vgg16(pretrained=True).to(device)
    bind_model(model)

    if config.pause:
        nsml.paused(scope=locals())

    bTrainmode = False

    if config.mode == 'train':
        bTrainmode = True

        """ set loss and optim"""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr=0.0002, weight_decay=0.0005)

        """ Load data """
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        dataset = datasets.ImageFolder(os.path.join(DATASET_PATH, 'train', 'train_data'), transform=transform)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        """ Training Model"""
        
        for epoch in range(nb_epoch):
            total_batch = len(train_loader)

            model.train()

            for idx, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)

                preds = model(images)

                loss = criterion(preds, labels)
                acc = (torch.argmax(preds, 1).cpu().numpy() == labels).sum().float() / len(preds)

                model.zero_grad()
                loss.backward()
                optimizer.step()

                ## print
                if (idx+1) % 1 == 0:
                    print('Train Epoch [%d/%d], Iter [%d/%d], loss: %.4f, acc: %.4f'
                         %(epoch+1, nb_epoch, idx+1, total_batch, loss.item(), acc.item()))
                        
                
                nsml.report(summary=True, epoch=epoch, epoch_total=nb_epoch, loss=loss.item(), acc=acc.item())
            nsml.save(epoch)