import numpy as np
import tensorflow as tf
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def wasserstein_loss_generator(fake_output):
    return tf.reduce_mean(fake_output)

def wasserstein_loss_discriminator(real_output, fake_output):
    return tf.reduce_mean(real_output)+tf.reduce_mean(fake_output)*-1
# Weight dictionary mapping keywords to functions. 
gen_loss={'cce':generator_loss,'was':wasserstein_loss_generator}
dis_loss={'cce':discriminator_loss,'was':wasserstein_loss_discriminator}