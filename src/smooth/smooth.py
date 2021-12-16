import PIL.Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pydensecrf.densecrf as dcrf
from PIL import Image, ImageFilter
import cv2 as cv
from pydensecrf.utils import (
    unary_from_labels,
    create_pairwise_bilateral,
    create_pairwise_gaussian
)


def crf(img,
        seg,
        smooth_kernel=True,
        rgb_kernel=True,
        rgb_scale_param=20,
        smooth_param=3):
    # img - np.array, anno - np.array 2dim

    seg = np.where(seg > 0, 1, 0).astype(np.uint8)

    n_labels = 2
    d = dcrf.DenseCRF(img.shape[1] * img.shape[0], n_labels)
    u = unary_from_labels(seg, n_labels, gt_prob=0.7, zero_unsure=False)
    d.setUnaryEnergy(u)

    if smooth_kernel:
        feats_pws = create_pairwise_gaussian(sdims=(smooth_param, smooth_param), shape=img.shape[:2])
        d.addPairwiseEnergy(feats_pws, compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

    if rgb_kernel:
        sdims = (rgb_scale_param, rgb_scale_param)
        feats_b = create_pairwise_bilateral(sdims=sdims, schan=(13, 13, 13), img=img, chdim=2)
        d.addPairwiseEnergy(feats_b, compat=10, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

    # run Inference for 5 steps
    q = d.inference(5)

    # find out the most probable class for each pixel
    m = np.argmax(q, axis=0)

    return m.reshape((img.shape[0], img.shape[1]))


def pil_smooth(mask: PIL.Image.Image):
    return mask.filter(ImageFilter.ModeFilter(size=13))


def apply_segnet_mask(img, mask):
    img, mask = np.array(img), np.array(mask)

    if np.max(mask) <= 1:
        mask = mask.astype(int) * 255

    if np.max(img) <= 1:
        img = img.astype(int) * 255

    alpha = np.stack([mask], axis=2)
    seg_img_with_mask = np.concatenate([img, alpha], axis=2)
    result = Image.fromarray(seg_img_with_mask.astype(np.uint8))
    return result


def smooth_raster_lines(im, filter_radius, filter_size, sigma):
    smoothed = np.zeros_like(im)
    contours, hierarchy = cv.findContours(im, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
    hierarchy = hierarchy[0]
    for countur_idx, contour in enumerate(contours):
        len_ = len(contour) + 2 * filter_radius
        idx = len(contour) - filter_radius

        x = []
        y = []
        for i in range(len_):
            x.append(contour[(idx + i) % len(contour)][0][0])
            y.append(contour[(idx + i) % len(contour)][0][1])

        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        x_filt = cv.GaussianBlur(x, (filter_size, filter_size), sigma, sigma)
        x_filt = [q[0] for q in x_filt]
        y_filt = cv.GaussianBlur(y, (filter_size, filter_size), sigma, sigma)
        y_filt = [q[0] for q in y_filt]

        smooth = []
        for i in range(filter_radius, len(contour) + filter_radius):
            smooth.append([x_filt[i], y_filt[i]])

        smooth_contours = np.asarray([smooth], dtype=np.int32)
        color = (0, 0, 0) if hierarchy[countur_idx][3] > 0 else (255, 255, 255)
        cv.drawContours(smoothed, smooth_contours, 0, color, -1)

    return smoothed


def cv_smooth(img, kernel_size=11):
    """Expected image in uint8 format (0 - 255) - NOT (0 - 1)!"""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    thresh, bin_red = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=3)
    return opening


if __name__ == '__main__':
    seg_mask = Image.open('mask.png')
    image = Image.open('img.png').convert('RGB')
    array_image, array_mask = np.array(image), np.array(seg_mask)

    opn = cv_smooth(array_mask, kernel_size=20)
    cvm = smooth_raster_lines(array_mask, 10, 5, 0.6)
    sm = crf(array_image, array_mask)
    pl = pil_smooth(seg_mask)

    default_result = apply_segnet_mask(array_image, array_mask)
    crf_resutl = apply_segnet_mask(array_image, sm)
    pil_result = apply_segnet_mask(array_image, pl)
    cv_result = apply_segnet_mask(array_image, cvm)
    opening_result = apply_segnet_mask(array_image, opn)

    fig, axs = plt.subplots(2, 2)
    axs[0][0].imshow(default_result)
    axs[0][0].set_title("Default")

    axs[0][1].imshow(crf_resutl)
    axs[0][1].set_title("CRF")

    axs[1][0].imshow(pil_result)
    axs[1][0].set_title("PIL ModeFilter")

    axs[1][1].imshow(opening_result)
    axs[1][1].set_title("CV2 Opening Morph.")

    # [ax.axis('off') for row in axs for ax in row]

    plt.show()
