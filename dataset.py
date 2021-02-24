import numpy as np
import os
import tensorflow as tf
import fnmatch
import _pickle as pickle    #cPickle
import utils.korean_manager as korean_manager
import random
import tensorflow_datasets as tfds
import progressbar

class DataLoader():
    #Load data patch by patch
    def __init__(self,data_path='./data',data_type='cifar10',batch_size=128):
        self.data_path = os.path.join(data_path,data_type)
        if not os.path.exists(self.data_path):
            os.path.mkdir(self.data_path)
        
        if data_type == 'cifar10':
            ds=tfds.load('cifar10',data_dir=self.data_path)['train']
            self.get_train=self.load_dataset_batch
            arr=[]
            for image in tqdm(ds):
                num_images+=1
                arr.append(image['image'])

            self.dataset=tf.data.Dataset.from_tensor_slices(arr).shuffle().batch(batch_size)
            self.iterator=iter(self.dataset)
        elif data_type == 'celeb_a'
            ds=tfds.load('celeb_a',data_dir=self.data_path)['train']
            self.get_train=self.load_dataset_batch
            arr=[]
            for image in tqdm(ds):
                num_images+=1
                arr.append(image['image'])

            self.dataset=tf.data.Dataset.from_tensor_slices(arr).shuffle().batch(batch_size)
            self.iterator=iter(self.dataset)

    def load_dataset_batch(self,path):
        optional = iterator.get_next_as_optional()
        if optional.has_value():
            return optional.get_value()
        else:
            #Reset iterator
            self.iterator=iter(self.dataset)
            return None
