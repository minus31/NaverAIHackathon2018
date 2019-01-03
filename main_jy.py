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

    def load(dir_name):
        print(dir_name)
        model.load_state_dict(os.path.join(dir_name, "model.pth"))
        print('model loaded!')

    def infer(queries, db):
        # Query 개수: 195
        # Reference(DB) 개수: 1,127
        # Total (query + reference): 1,322

        queries, query_img, references, reference_img = preprocess(queries, db)

        print('test data load queries {} query_img {} references {} reference_img {}'.
              format(len(queries), len(query_img), len(references), len(reference_img)))
        print(type(reference_img))

        # get image features

        


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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.resnet = models.vgg16(pretrained=False)
        self.resnet.fc = nn.Linear(in_features=2048, out_features=1000, bias=True)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.resnet(x)
        out = self.softmax(x)
        return out


if __name__ == '__main__':
    args = argparse.ArgumentParser()

    # hyperparameters
    args.add_argument('--epochs', type=int, default=2)
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = models.vgg16(pretrained=False).to(device)
    bind_model(model)
    nsml.save(0)
    nsml.load(checkpoint='0')

    if config.pause:
        nsml.paused(scope=locals())

    bTrainmode = False

    if config.mode == 'tarain':
        bTrainmode = True

        """ set loss and optim"""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.resnet.fc.parameters(), lr=0.0002, weight_decay=0.0005)

        """ Load data """
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        dataset = datasets.ImageFolder(os.path.join(DATASET_PATH, 'train', 'train_data'), transform=transform)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        """ Training Model"""
        nsml.load(checkpoint='0')

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