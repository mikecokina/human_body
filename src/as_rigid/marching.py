import numpy as np
import cv2

from matplotlib import pyplot as plt

im = cv2.imread('img.png', cv2.IMREAD_UNCHANGED)
imgray = cv2.cvtColor(im, cv2.COLOR_BGRA2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, 0)
im2, contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

plt.imshow(im2)
plt.show()


