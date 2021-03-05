import os
import argparse
import tensorflow_datasets as tfds
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import wandb
import utils.losses import L
import utils.model_architecture as Models
import evaluate
import datetime

class DCGAN():
    def __init__(self,gen_weights='',dis_weights='',generator='DCGAN64',discriminator='DCGAN64',noise_dim=100):
        self.noise_dim=noise_dim

        preprocessing_models={'DCGAN64':lambda :Models.build_input(),'DCGAN32':lambda :Models.build_input((32,32))}
        self.preprocessing=preprocessing_models[discriminator]()
        if gen_weights=='' and dis_weights=='':
            #Build model when weight path is not given
            generator_models={'DCGAN64':Models.build_generator64,'DCGAN32':Models.build_generator32}
            discriminator_models={'DCGAN64':Models.build_discriminator64,'DCGAN32':Models.build_discriminator32}

            self.generator = generator_models[generator]()
            self.discriminator = discriminator_models[discriminator]()
        else:
            #Load model
            if not gen_weights=='':
                self.generator = tf.keras.models.load_model(gen_weights,compile=False)
            if not dis_weights=='':
                self.discriminator = tf.keras.models.load_model(dis_weights,compile=False)

            self.noise_dim=self.generator.input.shape[-1]

    def calculate_gen_loss(self,fake_output):
        if loss=='cce':
            g=L.generator_loss(fake_output)
        elif loss=='was':
            g=L.wasserstein_loss_generator(fake_output)
        elif loss='wasgp':
            g=L.wasserstein_loss_generator(fake_output)
    def calculate_dis_loss(self,real_output, fake_output, real_images, fake_images):
        if loss=='cce':
            g=L.discriminator_loss(real_output,fake_output)
        elif loss=='was':
            g=L.wasserstein_loss_discriminator(real_output,fake_output)
        elif loss='wasgp':
            g=L.wasserstein_loss_discriminator(real_output,fake_output) + self.gp_ratio * L.gradient_penalty(real_images, fake_images, self.discriminator)
    @tf.function
    def train_step(self,images):
        logs={}
        noise = tf.random.normal([self.batch_size, self.noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)
            real_images = self.preprocessing(images)

            real_output = self.discriminator(real_images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = self.calculate_gen_loss(fake_output)
            disc_loss = self.calculate_dis_loss(real_output, fake_output, real_images, generated_images)

            logs['g_loss']=gen_loss
            logs['d_loss']=disc_loss

            gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

            self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
            self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        return logs

    def generate_and_save_images( self,epoch, test_input):
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

    def save_model(self):
        dir='./logs'
        self.generator.save(os.path.join(dir,'generator.h5'))
        self.discriminator.save(os.path.join(dir,'discriminator.h5'))

    def write_tensorboard(self,summary_writer,history,step):
        with summary_writer.as_default():
            for key in history.keys():
                tf.summary.scalar(key, history[key], step=step)

    def initialize_wandboard(self):
        wandb.init(project="DCGAN", config={ })

    def write_wandboard(self,logs):
        wandb.log(logs)

    def build_optimizer(self,optimizer,lr_gen,lr_dis):
        if optimizer=='sgd':
            self.generator_optimizer = tf.keras.optimizers.SGD(lr_gen)
            self.discriminator_optimizer = tf.keras.optimizers.SGD(lr_dis)
        elif optimizer=='adam':
            self.generator_optimizer = tf.keras.optimizers.Adam(lr_gen)
            self.discriminator_optimizer = tf.keras.optimizers.Adam(lr_dis)
        elif optimizer=='adabound':
            self.generator_optimizer = tf.keras.optimizers.Adam(lr_gen)
            self.discriminator_optimizer = tf.keras.optimizers.Adam(lr_dis)

    def train(self,dataset,epochs=10,lr_gen=0.001,lr_dis=0.001,batch_size=128,optimizer='adabound',loss='cce',evaluate_FID=True,
            evaluate_IS=True,generate_image=True,log_wandb=False,log_tensorboard=True,log_name='DCGAN-tensorflow',initialize_wandboard=False,
            log_times_in_epoch=10,num_samples=1000):
        #Initialize parameters for training
        self.batch_size=batch_size
        self.loss=loss
        logs={}
        log_period=dataset.image_num//(log_times_in_epoch*batch_size)
        self.build_optimizer(optimizer,lr_gen,lr_dis)

        summary_writer = tf.summary.create_file_writer("./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        if initialize_wandboard:
            self.initialize_wandboard()
        plot_noise = tf.random.normal([16, self.noise_dim])
        for epoch in range(epochs):
            start = time.time()
            count=0
            isEnd=False
            pbar = tqdm(range(dataset.image_num//batch_size + 1))
            while True:
                count+=1
                pbar.update(1)
                image_batch=dataset.get_train(self.batch_size)
                # Check end of dataset
                if image_batch==None:
                    break

                history=self.train_step(image_batch)

                if count%log_period==0:
                    #Log current state to wandb, plot smaples...
                    logs['G_loss']=history['g_loss']
                    logs['D_loss']=history['d_loss']
                    
                    image_batch=dataset.load_dataset_patch(num_samples,is_random=False)
                        
                    if evaluate_FID:
                        FID=evaluate.get_FID(self.generator,image_batch)
                        logs['FID']=FID
                        print('FID Score:',FID)
                    
                    if log_tensorboard:
                        self.write_tensorboard(summary_writer,logs,epoch)
                    if log_wandb:
                        self.write_wandboard(logs)
                    #Log Complete
            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
            # Plot sample images every epoch
            if generate_image:
                self.generate_and_save_images(epoch + 1,plot_noise)
        self.save_model()
