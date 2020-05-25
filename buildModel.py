import keras
import tensorflow as tf
from keras.backend import image_data_format
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

"""
Data parameters
Image Parameters
Batch parameters
Root directories for data
"""

image_width = 256
image_height = 256
batch_size = 16
epochs = 50
nb_train_samples=len(pd.read_csv(r'C:\Users\Alejandro\Documents\GitHub\MachineLearningModels\trainCsv.csv'))
nb_validation_samples=len(pd.read_csv(r'C:\Users\Alejandro\Documents\GitHub\MachineLearningModels\validationCsv.csv'))

training_directory = r'D:\DescargasChrome\data\train'
training_datagen = ImageDataGenerator(rescale=1. / 255,
                                      shear_range=0.2,
                                      zoom_range=0.2,
                                      rotation_range=20,
                                      width_shift_range=0.2,
                                      height_shift_range=0.2,
                                      horizontal_flip=True)

training_generator = training_datagen.flow_from_directory(training_directory,
                                                          batch_size=batch_size,
                                                          class_mode='categorical',
                                                          target_size=(image_height, image_width))

validation_directory = r'D:\DescargasChrome\data\validation'
validation_datagen = ImageDataGenerator(rescale=1. / 255,
                                        shear_range=0.2,
                                        zoom_range=0.2,
                                        rotation_range=20,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        horizontal_flip=True)

validation_generator = validation_datagen.flow_from_directory(validation_directory,
                                                              batch_size=batch_size,
                                                              class_mode='categorical',
                                                              target_size=(image_height, image_width))

if image_data_format() == 'channels_first':
    input_shape = (1, image_width, image_height) #1 means B&W images if it was RGB it will be 3
else:
    input_shape = (image_width, image_height, 1)

def build_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    model.fit_generator(
        training_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size)

    model.save_weights('first_try.h5')