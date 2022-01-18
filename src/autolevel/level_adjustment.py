from typing import Tuple

import cv2
import numpy as np
from matplotlib import pyplot as plt

BLACK_POINT = 0
OUTPUT_BLACK = 0
WHITE_POINT = 255
OUTPUT_WHITE = 255
MIDTONE_POINT = 128


def histogram(img) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts, bins = np.histogram(img, bins=range(0, 256))
    centroids = (bins[1:] + bins[:-1]) / 2
    return counts.astype(float), bins.astype(float), centroids.astype(float)


def get_clip_masks(image, black_point, white_point):
    highligh_mask = image > white_point
    shadow_mask = image < black_point
    return shadow_mask, highligh_mask


def clip(image, black_point, white_point):
    linear_transform = (255 * ((image - black_point) / (white_point - black_point))).astype(np.uint8)
    return linear_transform


def apply_gamma(image, gamma):
    return (255.0 * (np.power((image / 255), 1.0 / gamma))).astype(np.uint8)


def apply_clip_mask(image, shadow_mask, highligh_mask):
    image[highligh_mask] = 255
    image[shadow_mask] = 0
    return image


def get_gamma(midtone_point):
    gamma = 1
    midtone_normal = midtone_point / 255

    if midtone_point < 128:
        midtone_normal *= 2
        gamma = 1 + (9 * (1 - midtone_normal))
        gamma = np.min([gamma, 9.99])
    elif midtone_point > 128:
        midtone_normal = (midtone_normal * 2) - 1
        gamma = 1 - midtone_normal
        gamma = np.max([gamma, 0.01])
    return gamma


def apply_output_levels(image, output_shadow, output_highlight):
    """
    Scalling pixels within interval [output_shadow, output_highlight]
    """
    return ((image / 255) * (output_highlight - output_shadow) + output_shadow).astype(np.uint8)


def main():
    # get image
    path = "/home/mike/Instyle/lama_v2/dirs/data_root_dir/_val_source/01.jpg"
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # histogram
    # counts, bins, centroids
    input_historgram = list(histogram(img))
    input_historgram[0] /= int(np.max(input_historgram[0]))

    # gamma
    gamma = get_gamma(MIDTONE_POINT)
    gamma = 0.5

    # adjust levels
    shadow_mask, highligh_mask = get_clip_masks(img, BLACK_POINT, WHITE_POINT)
    img = clip(img, black_point=BLACK_POINT, white_point=WHITE_POINT)
    img = apply_clip_mask(img, shadow_mask, highligh_mask)
    img = apply_gamma(img, gamma=gamma)

    # output_histogram = histogram(img)
    img = apply_output_levels(img, 100, OUTPUT_WHITE)

    fig, axs = plt.subplots(1, 2, figsize=(9, 5))
    axs[0].clear()
    axs[0].imshow(img)

    axs[1].axvline(x=BLACK_POINT, color='b')
    axs[1].axvline(x=WHITE_POINT, color='r')
    axs[1].axvline(x=MIDTONE_POINT, color='g')
    axs[1].hist(
        input_historgram[2],
        bins=len(input_historgram[0]),
        weights=input_historgram[0],
        range=(min(input_historgram[1]), max(input_historgram[1])),
        color="k"
    )
    axs[0].axis("off")
    plt.show()


if __name__ == '__main__':
    main()
