import cv2
import matplotlib
import numpy as np
from matplotlib import pyplot as plt

matplotlib.use('TkAgg')

a = np.array([
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0],
    [0, 1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 0, 0],
    [0, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0]
], dtype=np.uint8)


b = np.array([
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0]
], dtype=np.uint8)

acmp = np.logical_not(a).astype(np.uint8)


x = np.array([
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0]
], dtype=np.uint8)


fig, ax = plt.subplots(1, 3)

ax[0].imshow(a)
ax[1].imshow(acmp)

y = cv2.morphologyEx(x, cv2.MORPH_DILATE, b, iterations=1)
y = np.logical_and(y, acmp).astype(np.uint8)

while not np.all(y == x):
    x = y
    y = cv2.morphologyEx(x, cv2.MORPH_DILATE, b, iterations=1)
    y = np.logical_and(y, acmp).astype(np.uint8)

y = np.logical_or(y, a)
ax[2].imshow(y)
plt.show()
