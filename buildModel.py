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
from tensorflow.keras.optimizers import RMSprop


"""
Data parameters
Image Parameters
Batch parameters
Root directories for data
"""

image_width = 256
image_height = 256
batch_size = 32
epochs = 50
nb_train_samples = len(pd.read_csv(r'C:\Users\Alejandro\Documents\GitHub\MachineLearningModels\trainCsv.csv'))
nb_validation_samples = len(pd.read_csv(r'C:\Users\Alejandro\Documents\GitHub\MachineLearningModels\validationCsv.csv'))
print(nb_train_samples, nb_validation_samples)
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
    input_shape = (3, image_width, image_height)  # 1 means B&W images if it was RGB it will be 3
else:
    input_shape = (image_width, image_height, 3)


def build_model():
    model = keras.models.Sequential()

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(256, 256, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(14, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(lr=0.001),
                  metrics=['acc'])

    history = model.fit(
        training_generator,
        steps_per_epoch=36,
        epochs=100,
        validation_data=validation_generator,
        validation_steps=36
    )

    model.save_weights('first_try_weights.h5')
    model.save('first_try.h5')


build_model()
