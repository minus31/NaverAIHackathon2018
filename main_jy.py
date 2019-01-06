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
        model.eval()

        transforms_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])            
            ])

        def hamming2(s1, s2):
            """Calculate the Hamming distance between two bit strings"""
            assert len(s1) == len(s2)
            return sum(c1 != c2 for c1, c2 in zip(s1, s2))

        def l2_distance(a1, a2):
            return np.sqrt(np.sum(np.power(a1 - a2, 2)))

        # get feature of queries
        dict_query_features = {}
        for path, img in zip(queries, query_img):
            img_pre = transforms_test(img)
            output = model(img_pre.cuda().unsqueeze(0))
            
            ## l2
            dict_query_features[path.split('/')[-1].split('.')[0]] = output.cpu().detach().numpy()

            ## hamming
            ##dict_query_features[path.split('/')[-1].split('.')[0]] = "".join((output > 0).cpu().detach().numpy().astype('str'))

        # get feature of references
        dict_reference_features = {}
        for path, img in zip(references, reference_img):
            img_pre = transforms_test(img)
            output = model(img_pre.cuda().unsqueeze(0))

            ## l2
            dict_reference_features[path.split('/')[-1].split('.')[0]] = output.cpu().detach().numpy()

            ## hamming
            ##dict_reference_features[path.split('/')[-1].split('.')[0]] = "".join((output > 0).cpu().detach().numpy().astype('str'))

        # make list
        li_result = []

        for query_key, query_features in dict_query_features.items():
            dict_hamming = {}
            for reference_key, reference_features in dict_reference_features.items():
                ## l2
                dict_hamming[reference_key] = l2_distance(query_features, reference_features)
                ## hamming
                ## dict_hamming[reference_key] = hamming2(query_features, reference_features)
            
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
import torch
import torch.nn as nn
import torch.optim as optim 
from torch.utils.data import DataLoader
import torchvision
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as utils
import torch.nn.functional as F

# function
def make_triplet_set(path_data, k=10):
    def make_class_file(path_data):
        li_class = [x for x in os.listdir(path_data) if x != '.DS_Store']

        li_class_filename = {}
        for classname in li_class:
            li_class_filename[classname] = [classname + '/' + filename for filename in os.listdir(os.path.join(path_data, classname)) if filename.endswith('.jpg')]

        return li_class_filename
    
    class_file = make_class_file(path_data)
    
    li_result = []
    li_class = class_file.keys()
    for k, v in class_file.items():
        for _ in range(10):
            triplet_set = list(np.random.choice(class_file[k], 2, replace=False))

            negative_idx = np.random.choice(list(li_class), 1)
            while negative_idx[0] == k:
                negative_idx = np.random.choice(list(li_class), 1)

            negative_filename = np.random.choice(class_file[negative_idx[0]], 1)[0]
            triplet_set.append(negative_filename)
            
            li_result.append(triplet_set)
    
    print(li_result[0])
    return li_result

def get_image_from_path(path):
    img = cv2.imread(path, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    return img

transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

# DataLoader
class TripletImageSet(torch.utils.data.Dataset):
    def __init__(self, path_image, k, transform=None, loader=None):
        
        self.path = path_image
        self.triplet_set = make_triplet_set(self.path, k=20)
        self.transform = transform
        self.loader = loader
        
    def __getitem__(self, index):
        path1, path2, path3 = self.triplet_set[index]
        img1 = self.loader(os.path.join(self.path, path1))
        img2 = self.loader(os.path.join(self.path, path2))
        img3 = self.loader(os.path.join(self.path, path3))
        
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
            
        return img1, img2, img3
    
    def __len__(self):
        return len(self.triplet_set)

# Model
class Tripletnet(nn.Module):
    def __init__(self, embeddingnet):
        super(Tripletnet, self).__init__()
        self.embeddingnet = embeddingnet

    def forward(self, x, y, z):
        
        embedded_x = self.embeddingnet(x)
        embedded_y = self.embeddingnet(y)
        embedded_z = self.embeddingnet(z)
        
        dist_a = F.pairwise_distance(embedded_x, embedded_y, 2)
        dist_b = F.pairwise_distance(embedded_x, embedded_z, 2)
        return dist_a, dist_b, embedded_x, embedded_y, embedded_z

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.model = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(2048, 1000)
        
    def forward(self, x):
        output = self.model(x)
        output = self.fc(output.squeeze())
        return output



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

    embeddingnet = Net().to(device)
    bind_model(embeddingnet)
    tnet = Tripletnet(embeddingnet).to(device)

    if config.pause:
        nsml.paused(scope=locals())

    bTrainmode = False

    if config.mode == 'train':
        bTrainmode = True

        """ set loss and optim"""
        criterion = torch.nn.MarginRankingLoss(margin = 0.1)
        optimizer = optim.SGD(tnet.embeddingnet.fc.parameters(), lr=0.001, weight_decay=0.0005, momentum=0.00005)

        """ Load data """
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image_set = TripletImageSet(DATASET_PATH + '/train/train_data', 20, transform=transform, loader=get_image_from_path)
        loader = DataLoader(image_set, batch_size=batch_size, shuffle=True)
        """ Training Model"""
        
        for epoch in range(nb_epoch):
            total_batch = len(image_set)

            tnet.train()

            for idx, (img1, img2, img3) in enumerate(loader):
                img1 = img1.to(device)
                img2 = img2.to(device)
                img3 = img3.to(device)

                dista, distb, embedded_x, embedded_y, embedded_z = tnet(img1, img2, img3)
                target = torch.FloatTensor(dista.size()).fill_(1)
                target = target.to(device)
                
                loss_triplet = criterion(dista, distb, target)
                #loss_embedd = embedded_x.norm(2) + embedded_y.norm(2) + embedded_z.norm(2)
                loss = loss_triplet

                # compute gradient and do optimizer step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                ## print
                if (idx+1) % 10 == 0:
                    print(loss_triplet, dista, distb)
                    print('Train Epoch [%d/%d], Iter [%d/%d], loss_triplet: %.4f'
                         %(epoch+1, nb_epoch, idx+1, total_batch, loss_triplet.item(), ))
                        
                
                nsml.report(summary=True, epoch=epoch, epoch_total=nb_epoch, loss=loss.item(), loss_triplet=loss_triplet.item())
            
            nsml.save(epoch)
