import numpy as np
import os
import tensorflow as tf
import fnmatch
import _pickle as pickle    #cPickle
import random
import tensorflow_datasets as tfds
from tqdm import tqdm

class DataLoader():
    #Load data patch by patch
    def __init__(self,data_path='./data',data_type='cifar10',batch_size=128):
        self.data_path = os.path.join(data_path,data_type)
        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)
        
        if data_type == 'cifar10':
            print('Downloading CIFAR10 data')
            ds=tfds.load('cifar10',data_dir=self.data_path)['train']
            self.get_train=self.load_dataset_batch

            print('Processing CIFAR10 data')
            self.image_num=0
            arr=[]
            for image in tqdm(ds):
                self.image_num+=1
                arr.append(image['image'])

            self.dataset=tf.data.Dataset.from_tensor_slices(arr).shuffle(self.image_num).batch(batch_size)
            self.iterator=iter(self.dataset)
        elif data_type == 'celeb_a':
            print('Downloading celeba data')
            ds=tfds.load('celeb_a',data_dir=self.data_path)['train']
            self.get_train=self.load_dataset_batch

            print('Processing celeba data')
            self.image_num=0
            arr=[]
            for image in tqdm(ds):
                self.image_num+=1
                arr.append(image['image'])

            self.dataset=tf.data.Dataset.from_tensor_slices(arr).shuffle(self.image_num).batch(batch_size)
            self.iterator=iter(self.dataset)

    def load_dataset_batch(self,path):
        optional = iterator.get_next_as_optional()
        if optional.has_value():
            return optional.get_value()
        else:
            #Reset iterator
            self.iterator=iter(self.dataset)
            return None
