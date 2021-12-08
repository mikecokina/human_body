import cv2
import numpy as np

import matplotlib.pyplot as plt

IMAGE_FILE = "img.png"
tps = cv2.createThinPlateSplineShapeTransformer()
img = cv2.imread(IMAGE_FILE, cv2.IMREAD_COLOR)

from_points = [[300.79870129870136, 140.409090909091],
               [872.227272727273, 146.9025974025974],
               [388.46103896103904, 1085.2142857142858],
               [843.0064935064936, 1098.2012987012988]]
to_points = [[427.422077922078, 153.3961038961038],
             [726.1233766233768, 202.0974025974026],
             [404.69480519480527, 939.1103896103897],
             [781.318181818182, 1085.2142857142858]]

from_pts = np.array(from_points, dtype=np.float32).reshape((1, -1, 2))
to_pts = np.array(to_points, dtype=np.float32).reshape((1, -1, 2))

matches = [cv2.DMatch(i, i, 0) for i in range(0, len(to_points))]

dst = np.zeros(img.shape)
tps.estimateTransformation(to_pts, from_pts, matches)
respic = tps.warpImage(img, borderMode=cv2.BORDER_CONSTANT)

plt.axis("off")
plt.imshow(respic[::, ::, ::-1])
plt.show()
