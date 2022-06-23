#!/usr/bin/env python
# coding: utf-8
import tensorflow as tf
import numpy as np

training_data = np.load('DB_Files/npy_arr.npy', allow_pickle=True)
training_labels = np.load('DB_Files/bln_arr.npy')

eps = 5
lys = 64

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(108, 192)),
    tf.keras.layers.Dense(lys, activation=tf.nn.relu),
    tf.keras.layers.Dense(4, activation=tf.nn.softmax)
])


model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_data, training_labels, epochs=eps)

model.evaluate(training_data, training_labels)

tf.keras.models.save_model(filepath=f'Models/{lys}-{eps}', model=model)
