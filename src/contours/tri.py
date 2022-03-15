import cv2
import numpy as np
import matplotlib.pyplot as plt
import triangle as tr

# face = tr.get_data('face')
# t = tr.triangulate(face, 'p')
# tr.compare(plt, face, t)
# plt.show()

from shapely.geometry import Polygon

image = cv2.imread('../smooth/mask.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

x = np.array([x[0] for x in contours[0]]).T

z = np.random.randint(0, len(x.T) - 1, 10000)
uv = np.array([x[0][z], x[1][z]]).T

poly = Polygon(uv)
poly = poly.simplify(tolerance=20, preserve_topology=False)
x, y = poly.exterior.coords.xy
uv = np.array([x, y]).T

data = {"vertices": uv}

plt.scatter(uv.T[0], uv.T[1])
plt.show()
exit()

# data["vertices"] = [[0, 0], [0, 1], [1, 1], [1, 0]]
t = tr.triangulate(data, 'a10000q30')

vertices, tris = t["vertices"], t["triangles"]

# outliers = np.array([cv2.pointPolygonTest(contours[0], v, False) for v in vertices]) < 0
# tri_indices = np.arange(0, len(vertices))
# outliers_indices = tri_indices[outliers]
# relevant_tris = [tri for tri in tris if not np.any(np.in1d(outliers_indices, tri))]
#
# t["triangles"] = relevant_tris

tr.compare(plt, data, t)
plt.gca().invert_yaxis()
plt.show()
