import keras
import tensorflow as tf
from keras import optimizers
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
from keras.optimizers import SGD
from keras.callbacks import CSVLogger


"""
Data parameters
Image Parameters
Batch parameters
Root directories for data
"""
labels = ['Atelectasis', 'Consolidation', 'Infiltration',
       'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis', 'Effusion',
       'Pneumonia', 'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass',
       'Hernia']

image_width = 256
image_height = 256
batch_size = 32
epochs = 50
nb_train_samples = len(pd.read_csv(r'C:\Users\Alejandro\Documents\GitHub\MachineLearningModels\trainCsv.csv'))
nb_validation_samples = len(pd.read_csv(r'C:\Users\Alejandro\Documents\GitHub\MachineLearningModels\validationCsv.csv'))
print(nb_train_samples, nb_validation_samples)
dataframe_train = pd.read_csv(r'C:\Users\Alejandro\Documents\GitHub\MachineLearningModels\trainCsv.csv')
dataframe_validation = pd.read_csv(r'C:\Users\Alejandro\Documents\GitHub\MachineLearningModels\validationCsv.csv')
dataframe_test = pd.read_csv(r'C:\Users\Alejandro\Documents\GitHub\MachineLearningModels\testCsv.csv')
print(dataframe_train.head())
print(dataframe_train.columns)

datagen = ImageDataGenerator(rescale=1. / 255.)
test_datagen = ImageDataGenerator(rescale=1. / 255.)

train_generator = datagen.flow_from_dataframe(
    dataframe=dataframe_train,
    directory="D:\DescargasChrome\input\images*\images",
    x_col="FullPath",
    y_col=labels,
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="raw",
    target_size=(256, 256))
valid_generator = test_datagen.flow_from_dataframe(
    dataframe=dataframe_validation,
    directory="D:\DescargasChrome\input\images*\images",
    x_col="FullPath",
    y_col=labels,
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="raw",
    target_size=(256, 256))

test_generator = test_datagen.flow_from_dataframe(
    dataframe=dataframe_test,
    directory="D:\DescargasChrome\input\images*\images",
    x_col="FullPath",
    y_col=labels,
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="raw",
    target_size=(256, 256))
# predictions=0.5
# results=pd.DataFrame(predictions, columns=labels)
# print("test")
# results["Filenames"]=test_generator.filenames
# ordered_cols=["Filenames"]+labels
# results=results[ordered_cols]#To get the same column order
# results.to_csv("results.csv",index=False)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(256,256,3))) #change to 448x448x1
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128)) #512
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(14, activation='sigmoid'))
model.summary()
model.compile(optimizers.rmsprop(lr=0.0001, decay=1e-6),loss="binary_crossentropy",metrics=["accuracy"])

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size


model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=1
)
model.save('modelo2.h5')
model_json= model.to_json()
with open("model_num2.json", "w") as json_file:
    json_file.write(model_json)
    
test_generator.reset()
pred=model.predict_generator(test_generator,
    steps=STEP_SIZE_TEST,
    verbose=1)
pred_bool = (pred >0.5)

predictions = pred_bool.astype(int)

#columns should be the same order of y_col
results=pd.DataFrame(predictions, columns=labels)
print(len(test_generator.filenames))
results["Image Index"]=test_generator.filenames
ordered_cols=["FullPath"]+labels
results=results[ordered_cols]#To get the same column order
results.to_csv("results.csv",index=False)


# training_directory = r'D:\DescargasChrome\data\train'
# training_datagen = ImageDataGenerator(rescale=1. / 255,
#                                       shear_range=0.2,
#                                       zoom_range=0.2,
#                                       rotation_range=20,
#                                       width_shift_range=0.2,
#                                       height_shift_range=0.2,
#                                       horizontal_flip=True)
#
# training_generator = training_datagen.flow_from_directory(training_directory,
#                                                           batch_size=batch_size,
#                                                           class_mode='categorical',
#                                                           target_size=(image_height, image_width))
#
# validation_directory = r'D:\DescargasChrome\data\validation'
# validation_datagen = ImageDataGenerator(rescale=1. / 255,
#                                         shear_range=0.2,
#                                         zoom_range=0.2,
#                                         rotation_range=20,
#                                         width_shift_range=0.2,
#                                         height_shift_range=0.2,
#                                         horizontal_flip=True)
#
# validation_generator = validation_datagen.flow_from_directory(validation_directory,
#                                                               batch_size=batch_size,
#                                                               class_mode='categorical',
#                                                               target_size=(image_height, image_width))
#
# if image_data_format() == 'channels_first':
#     input_shape = (3, image_width, image_height)  # 1 means B&W images if it was RGB it will be 3
# else:
#     input_shape = (image_width, image_height, 3)
#

# def build_model(): firs try this should be useful for multiclass but is multilabel
#     model = keras.models.Sequential()
#
#     model = tf.keras.models.Sequential([
#         tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(256, 256, 3)),
#         tf.keras.layers.MaxPooling2D(2, 2),
#         tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
#         tf.keras.layers.MaxPooling2D(2, 2),
#         tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
#         tf.keras.layers.MaxPooling2D(2, 2),
#         tf.keras.layers.Flatten(),
#         tf.keras.layers.Dense(1024, activation='relu'),
#         tf.keras.layers.Dense(14, activation='softmax') #change for sigmoid if multilabel
#     ])
#     model.compile(loss='categorical_crossentropy',
#                   optimizer=RMSprop(lr=0.001),
#                   metrics=['acc'])
#
#     history = model.fit(
#         training_generator,
#         steps_per_epoch=36,
#         epochs=100,
#         validation_data=validation_generator,
#         validation_steps=36
#     )
#
#     model.save_weights('first_try_weights.h5')
#     model.save('first_try.h5')

# def build_model():
#     model = Sequential()
#     model.add(Conv2D(32, (3, 3), padding='same',
#                      input_shape=(256, 256, 3)))
#     model.add(Activation('relu'))
#     model.add(Conv2D(32, (3, 3)))
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))
#     model.add(Conv2D(64, (3, 3), padding='same'))
#     model.add(Activation('relu'))
#     model.add(Conv2D(64, (3, 3)))
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))
#     model.add(Flatten())
#     model.add(Dense(512))
#     model.add(Activation('relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(14, activation='sigmoid'))
#     model.compile(optimizers.rmsprop(lr=0.0001, decay=1e-6), loss="binary_crossentropy", metrics=["accuracy"])
#
# build_model()
