"""
Find closer transformation of existing points located on both pictures.
Let's assume we have image I (x, y) and second image J (w, h).
Denspose does a job and returns uv coordinates for 24 (i) preefined "skin" parts.
The job is done for both I and J.
The goal of following code is to find mapping from [x, y] -> [w, h] for coresponding uv coordinates.
The fact remains that uv for I and J will most likely never match, but the goal is to find mapping in
predefined thresholds, for example [x1, y1] = u(0.9), v(0.9) and [w1, h1] = u(0.909), u(0.899) is in some reasonable
threshold close enough and it means, [x1, y1] might be mapped to [w1, h1].
"""
import pickle
import cv2
import matplotlib
import numpy as np
from detectron2.data.detection_utils import read_image

from PIL import Image
from matplotlib import pyplot as plt
from scipy.spatial import distance_matrix
from predictor import DensposePredictor

matplotlib.use('TkAgg')


def vector_to_matrix_index(vector_index, cols) -> list:
    j = int(vector_index % cols)
    i = int((vector_index - j) / cols)
    return [i, j]


def resize_image(image, scale=1.0):
    instance_of = type(image)
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8))

    width, height = image.size
    newsize = (int(width * scale), int(height * scale))
    image = image.resize(newsize)

    if instance_of == np.ndarray:
        return np.array(image, dtype=np.float16)
    return image


def denspose2iuv(result):
    tmp = DensposePredictor.parse_vui(result).astype(np.float32)
    iuv = np.array([tmp[:, :, 2], tmp[:, :, 1], tmp[:, :, 0]], dtype=np.float32)
    return iuv


def im2im_mapper(im1, im2):
    im1 = Image.fromarray(im1.copy().astype(np.uint8))
    im2 = Image.fromarray(im2.copy().astype(np.uint8))

    x_scale, y_scale = np.array(im1.size) / np.array(im2.size)

    mapper = []
    for _y in range(im2.size[1]):
        # add line
        mapper.append([])
        for _x in range(im2.size[0]):
            mapper[-1].append([int(_x * x_scale), int(_y * y_scale)])

    return np.array(mapper, dtype=int)


def densepose_mapper(image_file_1, image_file_2, image_1_dst, image_2_dst, threshold=1):
    mapping = []

    p = DensposePredictor(config_file='/home/mike/Instyle/densepose/src/config/densepose_rcnn_R_50_FPN_s1x.yaml',
                          model='/home/mike/Instyle/densepose/src/model/model_final_162be9.pkl')

    image_1 = read_image(image_file_1).astype(np.float32)
    scaled_image_1 = resize_image(image_1, scale=0.15).astype(np.float32)

    image_2 = read_image(image_file_2).astype(np.float32)
    scaled_image_2 = resize_image(image_2, scale=0.15).astype(np.float32)

    denspose_1 = p.predict(image=scaled_image_1)
    dx_1, dy_1, _, _ = p.parse_bbox(denspose_1)
    dx_1, dy_1 = int(dx_1), int(dy_1)

    denspose_2 = p.predict(image=scaled_image_2)
    dx_2, dy_2, _, _ = p.parse_bbox(denspose_2)
    dx_2, dy_2 = int(dx_2), int(dy_2)

    iuv_1 = p.parse_vui(denspose_1).astype(np.float32)
    iuv_1 = np.array([iuv_1[:, :, 2], iuv_1[:, :, 1], iuv_1[:, :, 0]], dtype=np.float32)

    iuv_2 = p.parse_vui(denspose_2).astype(np.float32)
    iuv_2 = np.array([iuv_2[:, :, 2], iuv_2[:, :, 1], iuv_2[:, :, 0]], dtype=np.float32)

    i_uv = np.stack((iuv_1[1], iuv_1[2]), axis=2)
    j_uv = np.stack((iuv_2[1], iuv_2[2]), axis=2)

    i_height, i_width = i_uv.shape[:2]
    j_height, j_width = j_uv.shape[:2]

    i_uv = i_uv.reshape(i_width * i_height, 2)
    j_uv = j_uv.reshape(j_width * j_height, 2)

    ds = distance_matrix(i_uv, j_uv)

    # Itterate over parts.
    for part_index in range(1, 25):
        mapping.append([])

        # Get only those pixels that are relevant (belong to given part index.)
        i_part_pixels = (iuv_1[0] == part_index).reshape(-1)
        j_part_pixels = (iuv_2[0] == part_index).reshape(-1)

        if np.sum(i_part_pixels) == 0:
            continue
        # Create array of indices in i_uv and j_uv vector with positions related to given part (defined by part index).
        i_part_indices = np.arange(0, len(i_part_pixels))[(iuv_1[0] == part_index).reshape(-1)]
        # j_part_indices = np.arange(0, len(j_part_pixels))[(iuv_2[0] == part_index).reshape(-1)]

        for iuv_idx in i_part_indices:
            # Rows in ds matrix indicates indices (order) of points from i_uv vector.
            # Each position in row indicates uv distance from point in i_uv vector to given
            # (point defined by index in given row) in j_uv vector.
            # Example: let's say we have two vectors x = [x_p1, x_p2], y = [y_p1, y_p2, y_p3]
            #   Then distances in matrix ds are in following form:
            #       [[d(x_p1, y_p1), d(x_p1, y_p2), d(x_p1, y_p3)],
            #        [d(x_p2, y_p1), d(x_p2, y_p2), d(x_p2, y_p3)]]

            # The `uv_closest_index` defines index of point from j_uv vector (constrained by `j_part_indices`) which
            # is closest to point from i_uv defined by row `iuv_idx`.
            j = ds[iuv_idx].copy()

            j[~j_part_pixels] = np.inf
            uv_closest_index = np.argmin(j)
            if j[uv_closest_index] > threshold:
                continue
            # Mapping tells which point from i_uv corresponding to the point in j_uv ( in order [i_uv, j_uv])
            mapping[part_index - 1].append([iuv_idx, uv_closest_index])
            # We need to set given column to higher value as possible to avoid mapping when two different i_uv points
            # will be mapped to the same j_uv point just because thery are closest to each other even though there
            # is already assigned point before.
            # This idea will most likely requires review and get better solution than such kind of random decisioning.
            ds[:, uv_closest_index] = np.inf

    # Remapping from vectior to original matrix.
    # The mapping from previous for loop gives as an information about mapping from vector i_uv to vector j_uv,
    # but we need to transform this mapping to actual x, y to w, h mapping (from x, y coordinates of one picture)
    # to (w, h) coordinates to another picture.
    i_width, j_width = i_width, j_width

    # Transform results to separate corresponing matrices.
    xy, wh = [], []
    for idx, part_mapping in enumerate(mapping):
        xy.append([])
        wh.append([])
        for xy_wh in part_mapping:
            xy[-1].append(vector_to_matrix_index(xy_wh[0], i_width))
            wh[-1].append(vector_to_matrix_index(xy_wh[1], j_width))

    mapaping_1_dst, mapaping_2_dst = f"{image_1_dst}.mapping", f"{image_2_dst}.mapping"

    xy.append([dx_1, dy_1]), wh.append([dx_2, dy_2])
    pickle.dump(xy, open(mapaping_1_dst, "wb"))
    pickle.dump(wh, open(mapaping_2_dst, "wb"))

    cv2.imwrite(image_1_dst, scaled_image_1[::, ::, ::-1])
    cv2.imwrite(image_2_dst, scaled_image_2[::, ::, ::-1])

    # # Plot data.
    # fig, axs = plt.subplots(1, 2)
    #
    # for i, part in enumerate(zip(xy, wh)):
    #     _xy, _wh = part
    #     _xy, _wh = np.array(_xy), np.array(_wh)
    #
    #     for a, b in zip(_xy, _wh):
    #         color = np.random.rand(3,).reshape(1, -1)
    #         # color1 = [scaled_image_1[dy_1 + a[0]][dx_1 + a[1]]/255]
    #         # color2 = [scaled_image_1[dy_1 + a[0]][dx_1 + a[1]] / 255]
    #         axs[0].scatter(a[1], a[0], c=color)
    #         axs[1].scatter(b[1], b[0], c=color)
    #
    #         # axs[0].scatter(i_dense[1][a[0]][a[1]], i_dense[0][a[0]][a[1]], c=color)
    #         # axs[1].scatter(j_dense[1][b[0]][b[1]], j_dense[0][b[0]][b[1]], c=color)
    #
    # axs[0].invert_yaxis()
    # axs[1].invert_yaxis()
    # plt.show()


if __name__ == '__main__':
    img1 = 'shirt_a_0.jpg'
    img2 = 'shirt_b_0.jpg'

    img1_dst = 'data/1.jpg'
    img2_dst = 'data/2.jpg'

    densepose_mapper(img1, img2, img1_dst, img2_dst)
