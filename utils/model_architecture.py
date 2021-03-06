from tensorflow.keras import layers
import tensorflow as tf

def build_generator64():
    model = tf.keras.Sequential()
    model.add(layers.Dense(4*4*1024, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((4,4,1024)))

    model.add(layers.Conv2DTranspose(512, 4, strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(256, 4, strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(128, 4, strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, 4, strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    return model

def build_discriminator64():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, 3, strides=(2, 2), padding='same',
                                     input_shape=(64,64,3)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(128, 3, strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(256, 3, strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(512, 3, strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model
    
def build_generator32():
    model = tf.keras.Sequential()
    model.add(layers.Dense(4*4*1024, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((4,4,1024)))

    model.add(layers.Conv2DTranspose(512, 4, strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(256, 4, strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(128, 4, strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, 4, strides=(1, 1), padding='same', use_bias=False, activation='tanh'))
    return model

def build_discriminator32():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, 3, strides=(1, 1), padding='same',
                                     input_shape=(32,32,3)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(128, 3, strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(256, 3, strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(512, 3, strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

def res_block_64(x,c,k_s,samp=None):
    if samp=='up':
        c1=layers.Conv2DTranspose(c,k_s,strides=(2, 2),padding='same',use_bias=False)
        c2=layers.Conv2D(c,k_s,padding='same',use_bias=False)
    elif samp=='down':
        c1=layers.Conv2D(c,k_s,strides=(2, 2), padding='same',use_bias=False)
        c2=layers.Conv2D(c,k_s, padding='same',use_bias=False)
    elif samp==None:
        c1=layers.Conv2D(c,k_s, padding='same',use_bias=False)
        c2=layers.Conv2D(c,k_s, padding='same',use_bias=False)

    sho
    return x

def build_discriminator64_wgangp():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(128, 3, strides=(2, 2), padding='same',input_shape=(32,32,3)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(128, 3, strides=(1, 1), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(128, 3, strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(128, 3, strides=(1, 1), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(128, 3, strides=(1, 1), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(128, 3, strides=(1, 1), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(128, 3, strides=(1, 1), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(128, 3, strides=(1, 1), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(1))

    return model

def build_input(image_size=(64,64)):
    reshape=tf.keras.models.Sequential([
      tf.keras.layers.experimental.preprocessing.Resizing(image_size[0],image_size[1],input_shape=(None,None,3)),
      tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5,offset=-1)
    ])
    return reshape
