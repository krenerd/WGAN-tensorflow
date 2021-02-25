import os
import argparse
import tensorflow_datasets as tfds
import tensorflow as tf
import progressbar
import matplotlib.pyplot as plt
import time

from dataset import DataLoader
import model
import evaluate

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
parser.add_argument("--samples_for_eval", type=int,default=1000)
parser.add_argument("--initial_epoch", type=int,default=0)
parser.add_argument("--epoch", type=int,default=100)
parser.add_argument("--evaluate_FID", type=str2bool,default=True)
parser.add_argument("--evaluate_IS", type=str2bool,default=True)
parser.add_argument("--dataset", type=str, choices=['celeba','cifar10'])
parser.add_argument("--generate_image", type=str2bool,default=True)

parser.add_argument("--gen_weights", type=str,default='')
parser.add_argument("--dis_weights", type=str,default='')

parser.add_argument("--loss", type=str, choices=['cce','was'])
parser.add_argument("--optimizer", type=str, choices=['adam','sgd','adabound'],default='adam')
parser.add_argument("--batch_size",type=int,default=64)
parser.add_argument("--learning_rate_dis",type=float,default=0.001)
parser.add_argument("--learning_rate_gen",type=float,default=0.001)

parser.add_argument("--generator", type=str, choices=['DCGAN64','DCGAN32'])
parser.add_argument("--discriminator", type=str, choices=['DCGAN64','DCGAN32'])
if __name__ == '__main__':
    args = parser.parse_args()
    
    dataset=DataLoader()
    GAN=model.DCGAN(gen_weights=args.gen_weights,dis_weights=args.dis_weights,generator=args.generator,discriminator=args.discriminator
        )
    GAN.train(dataset,epochs=args.epoch,lr_gen=args.learning_rate_gen,lr_dis=args.learning_rate_dis,batch_size=args.batch_size,optimizer=args.optimizer,
        loss=args.loss,evaluate_FID=args.evaluate_FID,evaluate_IS=args.evaluate_IS,generate_image=args.generate_image)