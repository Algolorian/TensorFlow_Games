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

# Greyscale, 25 epochs , 250 epochs      , 1000 epochs
# 64   accuracy: 0.5461,
# 128
# 256
# 512
# 1024
# 2048
# 8184

lys = 16384
eps = 25

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(108, 192)),
    tf.keras.layers.Dense(lys, activation=tf.nn.relu),
    tf.keras.layers.Dense(4, activation=tf.nn.softmax)
])

model.load_weights(f'Models/{lys}-{eps}')

model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.evaluate(training_data, training_labels)


# tf.keras.models.save_model(filepath=f'Models/{lys}-{eps}.tf', model=model)
