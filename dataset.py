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
    def __init__(self,data_path='./data',tfds_key='',batch_size=128):
        self.batch_size=batch_size
        if not tfds_key == '':
            print(f'Downloading {tfds_key} data')
            if data_path=='./data':
                data_path = os.path.join(data_path,tfds_key.split('/')[0])
                
            if not os.path.exists(data_path):
                os.mkdir(data_path)
            ds=tfds.load(tfds_key,data_dir=data_path)['train']

            self.get_train=self.load_dataset_batch

            print(f'Processing {tfds_key} data')
            self.image_num=0
            arr=[]
            for image in tqdm(ds):
                self.image_num+=1
                arr.append(image['image'])

            self.dataset=tf.data.Dataset.from_tensor_slices(arr)
            self.iterator=iter(self.dataset.shuffle(self.image_num).batch(batch_size))
            self.data_type='tfds'

    def load_dataset_batch(self,path):
        optional = self.iterator.get_next_as_optional()
        if optional.has_value():
            return optional.get_value()
        else:
            #Reset iterator
            self.iterator=iter(self.dataset.shuffle(self.image_num).batch(self.batch_size))
            return None

    def load_dataset_patch(self,num_samples,is_random=True):
        if self.data_type=='tfds':
            if is_random:
                ds=self.dataset.shuffle(self.image_num).batch(num_samples)
            else:
                ds=self.dataset.batch(num_samples)

            for patch in ds:
                return patch