import numpy as np
import cv2  # opencv-python
import matplotlib.pyplot as plt

img = cv2.imread('../pose/single.jpeg')
x = []
# get histogram for each channel
for i in cv2.split(img):
    hist, bins = np.histogram(i, 256, (0, 256))
    # discard colors at each end of the histogram which are used by only 0.05%
    use_color_indices = np.where(hist > hist.sum() * 0.0005)[0]
    i_min, i_max = use_color_indices.min(), use_color_indices.max()
    # stretch image colors by cliped histogram
    tmp = (i.astype(np.int32) - i_min) / (i_max - i_min) * 255
    tmp = np.clip(tmp, 0, 255)
    x.append(tmp.astype(np.uint8))

# combine image back and show it
s = np.dstack(x)
plt.imshow(s[::, ::, ::-1])
plt.show()
