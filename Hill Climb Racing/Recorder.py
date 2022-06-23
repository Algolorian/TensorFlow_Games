import numpy as np
import cv2
from mss import mss
import time

mon = {'left': 0, 'top': 0, 'width': 3840, 'height': 2160}
brk = {'left': 350, 'top': 1750, 'width': 350, 'height': 120}
gas = {'left': 3050, 'top': 1750, 'width': 350, 'height': 120}

img10 = cv2.imread('Images/1-0.png', cv2.IMREAD_UNCHANGED)
img20 = cv2.imread('Images/2-0.png', cv2.IMREAD_UNCHANGED)
img11 = cv2.imread('Images/1-1.png', cv2.IMREAD_UNCHANGED)
img21 = cv2.imread('Images/2-1.png', cv2.IMREAD_UNCHANGED)

with mss() as sct:
    i = 0
    inv = 0.1
    recorded_control = None
    while True:
        time.sleep(inv)

        img = np.array(sct.grab(mon))
        width = int(img.shape[1] / 20)
        height = int(img.shape[0] / 20)
        dim = (width, height)
        resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        resized_gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

        img = np.array(sct.grab(brk))
        width = int(img.shape[1] / 20)
        height = int(img.shape[0] / 20)
        dim = (width, height)
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        if np.array_equal(resized, img10):
            brk_off = True
        elif np.array_equal(resized, img11):
            brk_off = False
        else:
            recorded_control = None
            continue

        img = np.array(sct.grab(gas))
        width = int(img.shape[1] / 20)
        height = int(img.shape[0] / 20)
        dim = (width, height)
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        if np.array_equal(resized, img20):
            gas_off = True
        elif np.array_equal(resized, img21):
            gas_off = False
        else:
            recorded_control = None
            continue

        label = None
        if not brk_off and not gas_off:
            label = 3
        elif brk_off and not gas_off:
            label = 1
        elif gas_off and not brk_off:
            label = 2
        elif brk_off and gas_off:
            label = 0

        if recorded_control is None:
            recorded_control = np.array([label])
        else:
            recorded_control = np.concatenate((recorded_control, [label]), axis=0)
        # print(recorded_control)
        if len(recorded_control) == int(5/inv):
            # print(recorded_control)
            np.savetxt(f"""Recorded-control/record-{int(i-((5/inv)-1))}.csv""",
                       np.array([recorded_control[0]]), fmt='%s', delimiter=',')
            recorded_control = recorded_control[1:]

        cv2.imwrite(f'Recorded-images\\record-{i}.png', resized_gray_img)

        # print(recorded_control)
        np.save(f"""Recorded-DB/record-{i}""", resized_gray_img)

        i += 1
