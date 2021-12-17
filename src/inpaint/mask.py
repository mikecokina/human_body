import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


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


def gti_default():
    image_file = "git/in.png"
    mask_file = "gti/mask.png"
    out_file = "gti/out_in.png"

    seg_mask = Image.open(mask_file)
    seg_mask = np.array(seg_mask)
    if len(seg_mask.shape) > 2:
        seg_mask = np.array(seg_mask)[:, :, 0]

    image = Image.open(image_file)
    im_bool_mask = seg_mask == 255

    image = image.convert('RGB')
    image = np.array(image)
    image[::, ::, 0][im_bool_mask] = 255
    image[::, ::, 1][im_bool_mask] = 255
    image[::, ::, 2][im_bool_mask] = 255

    image = Image.fromarray(image)
    image.save(out_file)


def intern():
    image_file = "intern/in.png"
    mask_file = "intern/mask.png"
    out_file = "intern/out_in.png"
    out_mask = "intern/out_in_mask.png"

    seg_mask = Image.open(mask_file)
    seg_mask = np.array(seg_mask)
    if len(seg_mask.shape) > 2:
        seg_mask = np.array(seg_mask)[:, :, 0]

    image = Image.open(image_file).convert('RGB')
    image = np.array(image)

    seg_mask_bool = seg_mask == 0
    image[::, ::, 0][seg_mask_bool] = 255
    image[::, ::, 1][seg_mask_bool] = 255
    image[::, ::, 2][seg_mask_bool] = 255

    seg_mask = np.load('intern/mask.npy')

    max_vals = seg_mask == 255
    min_vals = seg_mask == 0
    one_vals = seg_mask == 1

    seg_mask[max_vals] = 255
    seg_mask[one_vals] = 0
    seg_mask[min_vals] = 0

    seg_out = Image.fromarray(seg_mask.astype(np.uint8))
    img_out = Image.fromarray(image)

    seg_out.save(out_mask)
    img_out.save(out_file)


def main():
    intern()


if __name__ == '__main__':
    main()
