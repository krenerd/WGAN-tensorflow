import os
import argparse
import tensorflow_datasets as tfds
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

from utils.losses import gen_loss,dis_loss
import utils.model_architecture as Models
import evaluate
import datetime

class DCGAN():
    def __init__(self,gen_weights='',disc_weights='',generator='DCGAN64',discriminator='DCGAN64',noise_dim=100):
        self.noise_dim=noise_dim

        self.preprocessing=Models.build_input()

        if weights=='' and disc_weights=='':
            #Build model when weight path is not given
            generator_models={'DCGAN64':Models.build_generator64,'DCGAN32':Models.build_generator32}
            discriminator_models={'DCGAN64':Models.build_discriminator64,'DCGAN32':Models.build_generator32}
            
            self.generator = generator_models[generator]
            self.discriminator = discriminator_models[discriminator]
        else:
            #Load model
            self.generator = tf.keras.models.load_model(gen_weights,compile=False)
            self.discriminator = tf.keras.models.load_model(disc_weights,compile=False)

            self.noise_dim=self.generator.input.shape[-1]

    @tf.function
    def train_step(images):
        logs={}
        noise = tf.random.normal([args.batch_size, noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(input_pipeline(images), training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = self.gen_loss_func(fake_output)
            disc_loss = self.dis_loss_func(real_output, fake_output)

            logs['g_loss']=gen_loss
            logs['d_loss']=disc_loss

            gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

            self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
            self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        return logs

    def generate_and_save_images( summary_writer,epoch, test_input):
        if not os.path.exists('./logs/images'):
            os.makedirs('./logs/images')

        predictions = self.generator(test_input, training=False)

        fig = plt.figure(figsize=(4,4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i+1)
            plt.imshow((predictions[i, :, :, :].numpy() * 127.5 + 127.5).astype(int))
            plt.axis('off')

        plt.savefig(f'./logs/images/epoch_{epoch}.png')
        plt.close()

        with summary_writer.as_default():
            tf.summary.image("Training data", plt.imread(f'./logs/images/epoch_{epoch}.png'), step=epoch)

    def save_model():
        dir='./logs'
        self.generator.save(os.path.join(dir,'generator.h5'))
        self.discriminator.save(os.path.join(dir,'discriminator.h5'))

    def write_tensorboard(summary_writer,history,step):
        with summary_writer.as_default():
            for key in history.keys():
                tf.summary.scalar(key, history[key], step=step)

    def build_optimizer(self,optimizer,lr_gen,lr_dis):
        if optimizer=='sgd':
            self.generator_optimizer = tf.keras.optimizers.SGD(lr_gen)
            self.discriminator_optimizer = tf.keras.optimizers.SGD(lr_dis)
        elif optimizer=='adam':
            self.generator_optimizer = tf.keras.optimizers.Adam(lr_gen)
            self.discriminator_optimizer = tf.keras.optimizers.Adam(lr_dis)
        elif optimizer='adabound':
            self.generator_optimizer = tf.keras.optimizers.Adam(lr_gen)
            self.discriminator_optimizer = tf.keras.optimizers.Adam(lr_dis)

    def train(self,dataset,epochs=10,lr_gen=0.001,lr_dis=0.001,batch_size=128,optimizer='adabound',loss='cce',evaluate_FID=True,
        evaluate_IS=True,generate_image=True):

        self.gen_loss_func,self.dis_loss_func=gen_loss[loss],dis_loss[loss]
        logs={'G_loss':[],'D_loss':[],'FID':[]}

        self.build_optimizer(optimizer,args.learning_rate_gen,args.learning_rate_dis)

        summary_writer = tf.summary.create_file_writer("./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        plot_noise = tf.random.normal([16, noise_dim])
        for epoch in range(args.epoch):
            start = time.time()

            isEnd=False
            while True:
                image_batch=dataset.get_train(args.batch_size)
                # Check end of dataset
                if image_batch==None:
                    break

                history=self.train_step(image_batch)
                logs['G_loss'].append(history['g_loss'])
                logs['D_loss'].append(history['d_loss'])

            # Produce images for the GIF as we go
            if generate_image:
                self.generate_and_save_images(summary_writer,epoch + 1,plot_noise)

            plot_losses(args,losses)
            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
            for image_batch in dataset.batch(args.samples_for_eval):
                if args.dataset=='celeba':
                    image_batch=image_batch['image']
                
                if args.evaluate_FID:
                    FID=get_FID(generator,image_batch)
                    losses['FID'].append(FID)
                    print('FID Score:',FID)
                break
        self.save_model()
