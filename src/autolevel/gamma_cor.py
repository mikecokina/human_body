import cv2
import numpy as np


# Somehow I found the value of `gamma=1.2` to be the best in my case
def adjust_gamma(image, gamma=1.2):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


_image = cv2.imread('../pose/single.jpeg')
_auto_result = adjust_gamma(_image)
cv2.imshow('auto_result', _auto_result)
cv2.waitKey()
