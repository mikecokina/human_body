import cv2
import numpy as np
import matplotlib.pyplot as plt
import triangle as tr

face = tr.get_data('face')
# t = tr.triangulate(face, 'p')
# tr.compare(plt, face, t)
# plt.show()

image = cv2.imread('../smooth/mask.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

x = np.array([x[0] for x in contours[0]]).T
z = np.random.randint(0, len(x.T) - 1, 500)
uv = np.array([x[0][z], x[1][z]]).T
data = {"vertices": uv}

# data["vertices"] = [[0, 0], [0, 1], [1, 1], [1, 0]]
t = tr.triangulate(data, 'a10000q30')
tr.compare(plt, data, t)
plt.gca().invert_yaxis()
plt.show()
