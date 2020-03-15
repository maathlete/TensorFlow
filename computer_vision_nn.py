#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 21:57:45 2020

@author: alexandra
"""

import tensorflow as tf
from tensorflow import keras
#import matplotlib.pyplot as plt

print(tf.__version__)

mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

#standardize:
training_images  = training_images / 255.0
test_images = test_images / 255.0

#viz:
#plt.imshow(train_images[2])
#print(train_labels[2])
#print(train_images[2])

#10 = len(set{labels})
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(), 
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])


model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=5)
print(model.evaluate(test_images, test_labels))

