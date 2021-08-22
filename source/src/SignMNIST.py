from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPool2D
from keras.preprocessing.image import ImageDataGenerator
from keras_preprocessing.image.utils import validate_filename
import tensorflow as tf
from tensorflow import keras
from keras import preprocessing, utils
import numpy as np
import pandas as pd
from tensorflow.python.ops.gen_array_ops import size
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

'''
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f',
            'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 
            'w', 'x', 'y', 'z'
            ]
#Load training data

data = tf.keras.utils.image_dataset_from_directory(
    'NewDataset/asl_dataset/', labels='inferred', label_mode='int', class_names=classes, color_mode='rgb', batch_size=32,
    image_size=(400, 400), shuffle=True, seed=0 
)
'''


train_data = pd.read_csv('SignMNIST/sign_mnist_train/sign_mnist_train.csv')
train_data

train_imgs = np.array(train_data.drop(columns=['label']))
train_imgs = np.array([np.reshape(i, (28,28)) for i in train_imgs])
train_imgs.shape

train_labels = train_data['label']

#Plot one of the training images make sure data is valid

import matplotlib.pyplot as plt

plt.imshow(train_imgs[0])
print(train_labels[0])

#Load test data

test_data = pd.read_csv('SignMNIST/sign_mnist_test/sign_mnist_test.csv')

test_imgs = np.array(test_data.drop(columns=['label']))
test_imgs = np.array([np.reshape(i, (28,28)) for i in test_imgs])

test_labels = test_data['label']

#Load the data into variables

x_train, y_train, x_test, y_test = train_imgs, train_labels, test_imgs, test_labels

x_train = x_train.astype('float32')
x_train /= 255.
x_test =x_test.astype('float32')
x_test /= 255.

#One hot encode the labels

y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

#Resolve error with dimensions

x_train = x_train.reshape(len(x_train), 28, 28, 1)
x_test = x_test.reshape(len(x_test), 28, 28, 1)

from tensorflow.keras import layers, models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), padding='valid'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='valid'))
model.add(layers.Dropout(0.4))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Dropout(0.4))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.4))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.4))
model.add(layers.Dense(25, activation = 'softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

model.fit(x_train, y_train, epochs = 10, validation_data=(x_test, y_test))

# model.save('SignModel2.h5')