#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 10:45:37 2022

@author: troyobernolte
"""

from skimage import data, color
from skimage.transform import rescale, resize
import matplotlib.pyplot as plt
import imageio as iio
import tensorflow as tf
import numpy as np
from tensorflow import keras
 
EPOCHS = 100
BATCH_SIZE = 128 # number of samples you feed in to your network at a time
VERBOSE = 1
NB_CLASSES = 10   # number of outputs = number of digits
N_HIDDEN = 128
VALIDATION_SPLIT = 0.2 # how much TRAIN is reserved for VALIDATION

mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)
# Feature engineering

RESHAPED = 784

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
input_shape = (28, 28, 1)

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalize inputs to be in [0, 1].

x_train /= 255
x_test /= 255

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')



# Build the model.
model = tf.keras.models.Sequential()

model.add(keras.layers.Dense(NB_CLASSES,
                                 input_shape=(RESHAPED,),
                                 name='dense_layer', 
                                 activation='softmax'))

    # Compiling the model.
model.compile(optimizer='SGD', 
              loss='categorical_crossentropy',
              metrics=['accuracy'])

def model_train():
    # Training the model.
    model.fit(x_train, y_train,
               batch_size=BATCH_SIZE, epochs=EPOCHS,
               verbose=VERBOSE, validation_split=VALIDATION_SPLIT)

    #evaluate the model
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print('Test accuracy:', test_acc)


# read an image
img = iio.imread("0-9-handwritten-5.jpg")
img_1 = img[100:200, 150:250], 5
img_2 = img[225:325, 150:250], 2
img_3 = img[340:440, 150:250], 3
img_4 = img[470:570, 150:250], 7
img_5 = img[580:680, 150:250], 4
img_6 = img[100:200, 310:410], 0
img_7 = img[225:325, 310:410], 1
img_8 = img[340:440, 310:410], 5
img_9 = img[470:570, 310:410], 2

images = [img_1, img_2, img_3, img_4, img_5, img_6, img_7, img_8, img_9]

def run_model():
    for num in images:
        plt.imshow(num[0])
        plt.show()
        image_resized = resize(num[0], (28, 28),
                               anti_aliasing=True)
        image_resized[image_resized == 1] = 0
        #image_resized  = image_resized/255.0
        #print(max(image_resized))
        image_resized  = np.array(image_resized)
        
        #reshaping to support our model input and normalizing
        image_resized  = image_resized .reshape(1,784)
        
        #predicting the class
        res = model.predict([image_resized])[0]
        #print(res, "Actual value:", num[1])
        print("Predicted value:", np.argmax(res), "Confidence:", max(res),
              "Actual value:", num[1], "\n")
        

