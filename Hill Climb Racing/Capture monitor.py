import numpy as np
import cv2
from mss import mss
import time

mon = {'left': 0, 'top': 0, 'width': 3840, 'height': 2160}

with mss() as sct:
    i = 0
    while True:
        img = np.array(sct.grab(mon))
        width = int(img.shape[1] / 10)
        height = int(img.shape[0] / 10)
        dim = (width, height)
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        cv2.imwrite(f'Images\\resized-{i}.png', resized)
        del img
        i += 1
        time.sleep(2)
