import numpy as np
import cv2
from mss import mss
import time

mon = {'left': 3050, 'top': 1750, 'width': 350, 'height': 120}

with mss() as sct:
    i = 0
    while True:
        img = np.array(sct.grab(mon))
        width = int(img.shape[1] / 20)
        height = int(img.shape[0] / 20)
        dim = (width, height)
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        cv2.imwrite(f'Images\\{i}.png', resized)
        i += 1
        time.sleep(2)

