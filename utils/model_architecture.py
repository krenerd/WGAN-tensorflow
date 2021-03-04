from tensorflow.keras import layers
import tensorflow as tf

def build_generator64(image_size=(64,64),channels=64):
    w,h=image_size
    
    model = tf.keras.Sequential()
    model.add(layers.Dense(w*h*channels//16, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((w//8,h//8,channels*4)))

    model.add(layers.Conv2DTranspose(channels*4, 4, strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(channels*2, 4, strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(channels, 4, strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Conv2DTranspose(3, 4, strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    return model

def build_discriminator64(image_size=(64,64),channels=64):
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(channels, 4, strides=(2, 2), padding='same',
                                     input_shape=(64,64,3)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    #model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(channels*2, 4, strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    #model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(channels*4, 4, strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(channels*8, 4, strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Flatten())
    model.add(layers.Dense(1,activation='sigmoid'))

    return model

def build_generator32():
    model = tf.keras.Sequential()
    model.add(layers.Dense(8*8*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((8,8,256)))

    model.add(layers.Conv2DTranspose(256, 4, strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(128, 4, strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, 4, strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, 4, strides=(1, 1), padding='same', use_bias=False, activation='tanh'))
    return model

def build_discriminator32():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, 4, strides=(2, 2), padding='same',
                                     input_shape=(32,32,3)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(128, 4, strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Flatten())
    model.add(layers.Dense(1,activation='sigmoid'))

    return model

def build_input(image_size=(64,64)):
    reshape=tf.keras.models.Sequential([
      tf.keras.layers.experimental.preprocessing.Resizing(image_size[0],image_size[1],input_shape=(None,None,3)),
      tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5,offset=-1)
    ])
    return reshape
