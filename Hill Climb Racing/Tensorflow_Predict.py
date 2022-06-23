#!/usr/bin/env python
# coding: utf-8
import tensorflow as tf
import numpy as np


training_data = np.load('DB_Files/npy_arr.npy', allow_pickle=True)
training_labels = np.load('DB_Files/bln_arr.npy')

print(list(training_labels).count(0))
print(list(training_labels).count(1))
print(list(training_labels).count(2))
print(list(training_labels).count(3))

# Greyscale,  25 epochs, 5 epochs
# 128   accuracy: 0.5461, accuracy: 0.5461
# 256   accuracy: 0.5461, accuracy: 0.7225
# 512   accuracy: 0.7221, accuracy: 0.3756
# 1024  accuracy: 0.6289, accuracy: 0.7039
# 2048  accuracy: 0.7068, accuracy: 0.6683
# 4096  accuracy: 0.5461, accuracy: 0.7109
# 8184  accuracy: 0.5461, accuracy: 0.7246
# 16384 accuracy: 0.5461, accuracy: 0.7112

lys = 8184
eps = 5

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(108, 192)),
    tf.keras.layers.Dense(lys, activation=tf.nn.relu),
    tf.keras.layers.Dense(4, activation=tf.nn.softmax)
])


model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(training_data, training_labels, epochs=eps)

tf.keras.models.save_model(filepath=f'Models/{lys}-{eps}', model=model)

model.evaluate(training_data, training_labels)
