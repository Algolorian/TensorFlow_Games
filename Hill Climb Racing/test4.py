#!/usr/bin/env python
# coding: utf-8
import tensorflow as tf
import numpy as np
import cv2
from mss import mss
import time
import pyautogui

mon = {'left': 0, 'top': 0, 'width': 3840, 'height': 2160}
brk = {'left': 350, 'top': 1750, 'width': 350, 'height': 120}

img10 = cv2.imread('Images/1-0.png', cv2.IMREAD_UNCHANGED)
img20 = cv2.imread('Images/2-0.png', cv2.IMREAD_UNCHANGED)
img11 = cv2.imread('Images/1-1.png', cv2.IMREAD_UNCHANGED)
img21 = cv2.imread('Images/2-1.png', cv2.IMREAD_UNCHANGED)

# Greyscale, 25 epochs , 250 epochs      , 1000 epochs
# 64
# 128
# 256
# 512
# 1024
# 2048
# 8184

lys = 8184
eps = 5

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(108, 192)),
    tf.keras.layers.Dense(lys, activation=tf.nn.relu),
    tf.keras.layers.Dense(4, activation=tf.nn.softmax)
])

model.load_weights(f'Models/{lys}-{eps}')

model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


with mss() as sct:
    i = 0
    choice = None
    past_choice = None
    while True:

        img = np.array(sct.grab(brk))
        width = int(img.shape[1] / 20)
        height = int(img.shape[0] / 20)
        dim = (width, height)
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        if np.array_equal(resized, img10):
            pass
        elif np.array_equal(resized, img11):
            pass
        else:
            choice = None
            if past_choice == 1:
                pyautogui.keyUp('right')
            elif past_choice == 2:
                pyautogui.keyUp('left')
            elif past_choice == 3:
                pyautogui.keyUp('left')
                pyautogui.keyUp('right')
            continue

        img = np.array(sct.grab(mon))
        width = int(img.shape[1] / 20)
        height = int(img.shape[0] / 20)
        dim = (width, height)
        resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        resized_gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

        predictions = model.predict(np.array([resized_gray_img]))
        np.set_printoptions(suppress=True)

        past_choice = choice
        choice = list(predictions[0, :]).index(max(predictions[0, :]))
        # print(choice)

        if past_choice != choice:
            print('**')
            print(choice)
            if past_choice == 1:
                pyautogui.keyUp('right')
            elif past_choice == 2:
                pyautogui.keyUp('left')
            elif past_choice == 3:
                pyautogui.keyUp('left')
                pyautogui.keyUp('right')
            if choice == 1:
                pyautogui.keyDown('right')
            elif choice == 2:
                pyautogui.keyDown('left')
            elif choice == 3:
                pyautogui.keyDown('left')
                pyautogui.keyDown('right')
