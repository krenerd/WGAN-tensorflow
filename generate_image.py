import os
import argparse
import tensorflow_datasets as tfds
import tensorflow as tf
import progressbar
import matplotlib.pyplot as plt
import numpy as np
import scipy
import model

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
#Define arguments 
parser = argparse.ArgumentParser(description='Download dataset')

def load_model():
    dir='./logs'
    g=tf.keras.models.load_model(os.path.join(dir,'generator.h5'))
    return g

if __name__ == '__main__':
    args = parser.parse_args()
    generator=load_model()
    
    noise=tf.random.normal([1, 100])
    predictions = generator(noise, training=False)
    
    plt.imshow((predictions[0].numpy()*127.5+127.5).astype(int))
    plt.axis('off')
    plt.savefig(f'./generated_image.png')