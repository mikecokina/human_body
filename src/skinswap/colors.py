import numpy as np

import cv2
import matplotlib.pyplot as plt

from skinswap import colors
from sklearn.cluster import KMeans


# import the necessary packages
def centroid_histogram(_clt):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    num_labels = np.arange(0, len(np.unique(_clt.labels_)) + 1)
    histogram, _ = np.histogram(_clt.labels_, bins=num_labels)
    # normalize the histogram, such that it sums to one
    histogram = histogram.astype("float")
    histogram /= histogram.sum()
    # return the histogram
    return histogram


def plot_colors(histogram, centroids):
    # initialize the bar chart representing the relative frequency
    # of each of the colors
    _bar = np.zeros((50, 300, 3), dtype="uint8")
    start_x = 0
    # loop over the percentage of each cluster and the color of
    # each cluster
    for (percent, color) in zip(histogram, centroids):
        # plot the relative percentage of each cluster
        end_x = start_x + (percent * 300)
        cv2.rectangle(_bar, (int(start_x), 0), (int(end_x), 50),
                      color.astype("uint8").tolist(), -1)
        start_x = end_x

    # return the bar chart
    return _bar


def main():
    pass


if __name__ == '__main__':
    main()


# construct the argument parser and parse the arguments


args = {
    "image": "01.jpg",
    "clusters": 5
}

# load the image and convert it from BGR to RGB so that
# we can dispaly it with matplotlib
image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

im = image.reshape((image.shape[0] * image.shape[1], 3))
clt = KMeans(n_clusters=args["clusters"])
clt.fit(im)

hist = centroid_histogram(clt)
bar = plot_colors(hist, clt.cluster_centers_)

# show our image
fig, axs = plt.subplots(1, 2)

axs[0].axis("off")
axs[0].imshow(image)
axs[1].imshow(bar)

plt.show()
