import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('../smooth/mask.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

x = np.array([x[0] for x in contours[0]]).T
z = np.random.randint(0, len(x.T) - 1, 1000)
bg = np.zeros((image.shape[0], image.shape[1]))
bg[x[1][z], x[0][z]] = 255
y = [560, 744]
bg[y[0], y[1]] = 255

uv = np.array([x[1][z], x[0][z]]).T
plt.imshow(bg)
plt.show()

# decetd wether pixel is insde of countour
r = cv2.pointPolygonTest(contours[0], y, False)
print(r)


# cv2.imshow('Contours', image)
# cv2.waitKey(0)


# # Let's load a simple image with 3 black squares
# image = cv2.imread('../smooth/img.png')
# cv2.waitKey(0)
#
# # Grayscale
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# # Find Canny edges
# edged = cv2.Canny(gray, 30, 200)
# # cv2.waitKey(0)
#
# # Finding Contours
# # Use a copy of the image e.g. edged.copy()
# # since findContours alters the image
# contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#
# # cv2.imshow('Canny Edges After Contouring', edged)
# # cv2.waitKey(0)
#
# print("Number of Contours found = " + str(len(contours)))
#
# # Draw all contours
# # -1 signifies drawing all contours
# cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
#
# cv2.imshow('Contours', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
