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
from keras.models import load_model

dataframe_train = pd.read_csv(r'C:\Users\Alejandro\Documents\GitHub\MachineLearningModels\trainCsv.csv')
dataframe_validation = pd.read_csv(r'C:\Users\Alejandro\Documents\GitHub\MachineLearningModels\validationCsv.csv')
dataframe_test = pd.read_csv(r'C:\Users\Alejandro\Documents\GitHub\MachineLearningModels\testCsv.csv')
labels = ['Atelectasis', 'Consolidation', 'Infiltration',
       'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis', 'Effusion',
       'Pneumonia', 'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass',
       'Hernia']
test_datagen = ImageDataGenerator(rescale=1. / 255.)
datagen = ImageDataGenerator(rescale=1. / 255.)
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
    batch_size=30,
    seed=42,
    shuffle=True,
    class_mode="raw",
    target_size=(256, 256))
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size

model = load_model('modelo.h5')
test_generator.reset()
pred=model.predict_generator(test_generator,
steps=STEP_SIZE_TEST,
verbose=1)
pred_bool = (pred >0.2)

predictions = pred_bool.astype(int)

#columns should be the same order of y_col
results=pd.DataFrame(predictions, columns=labels)
print(len(test_generator.filenames))
results["FullPath"]=test_generator.filenames
ordered_cols=["FullPath"]+labels
results=results[ordered_cols]#To get the same column order
results.to_csv("results.csv",index=False)
