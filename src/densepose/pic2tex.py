import copy

import cv2
import matplotlib
import torch
import numpy as np
from PIL import Image

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.engine import DefaultPredictor
from matplotlib import pyplot as plt

from densepose import add_densepose_config
from densepose.structures import DensePoseChartPredictorOutput
from densepose.vis.extractor import DensePoseResultExtractor, DensePoseOutputsExtractor
from densepose.structures import DensePoseEmbeddingPredictorOutput

CONFIG = 'densepose_rcnn_R_50_FPN_s1x.yaml'
MODEL = 'model_final_162be9.pkl'
IMAGE_FILE = 'img_2.png'
OPTS = []

matplotlib.use('TkAgg')


def setup_config():
    cfg = get_cfg()
    add_densepose_config(cfg)
    cfg.merge_from_file(CONFIG)
    cfg.merge_from_list(OPTS)
    cfg.MODEL.WEIGHTS = MODEL
    cfg.freeze()
    return cfg


def execute_on_outputs(outputs):
    image_fpath = IMAGE_FILE
    result = {"file_name": image_fpath}
    if outputs.has("scores"):
        result["scores"] = outputs.get("scores").cpu()
    if outputs.has("pred_boxes"):
        result["pred_boxes_XYXY"] = outputs.get("pred_boxes").tensor.cpu()
        if outputs.has("pred_densepose"):
            if isinstance(outputs.pred_densepose, DensePoseChartPredictorOutput):
                extractor = DensePoseResultExtractor()
            elif isinstance(outputs.pred_densepose, DensePoseEmbeddingPredictorOutput):
                extractor = DensePoseOutputsExtractor()
            else:
                raise Exception("Invalid instance.")
            result["pred_densepose"] = extractor(outputs)[0]
    return result


def concat_atlas_tex(given_tex):
    tex = None
    for i in range(0, 4):
        tex_tmp = given_tex[6 * i]
        for j in range(1 + 6 * i, 6 + 6 * i):
            tex_tmp = np.concatenate((tex_tmp, given_tex[j]), axis=1)
        if tex is None:
            tex = tex_tmp
        else:
            tex = np.concatenate((tex, tex_tmp), axis=0)
    return tex


def interpolate_tex(tex):
    valid_mask = np.array((tex.sum(0) != 0) * 1, dtype='uint8')
    radius_increase = 10
    kernel = np.ones((radius_increase, radius_increase), np.uint8)
    dilated_mask = cv2.dilate(valid_mask, kernel, iterations=1)
    region_to_fill = dilated_mask - valid_mask
    invalid_region = 1 - valid_mask
    actual_part_max = tex.max()
    actual_part_min = tex.min()
    actual_part_uint = np.array((tex - actual_part_min) / (actual_part_max - actual_part_min) * 255, dtype='uint8')
    actual_part_uint = cv2.inpaint(actual_part_uint.transpose((1, 2, 0)), invalid_region, 1,
                                   cv2.INPAINT_TELEA).transpose((2, 0, 1))
    actual_part = (actual_part_uint / 255.0) * (actual_part_max - actual_part_min) + actual_part_min
    # only use dilated part
    actual_part = actual_part * dilated_mask

    return actual_part


def get_texture(im, iuv, bbox, tex_part_size=200):
    # image expected to have shape [n_channels x width x height]
    # default loaded by cv2 is (h x w x ch)
    im = im.transpose(2, 1, 0) / 255

    # iuv expected to match source image 1:1
    image_w, image_h = im.shape[1], im.shape[2]

    bbox[2] = bbox[2] - bbox[0]
    bbox[3] = bbox[3] - bbox[1]
    x, y, w, h = [int(v) for v in bbox]
    bg = np.zeros((image_h, image_w, 3))
    bg[y:y + h, x:x + w, :] = iuv
    iuv = bg
    iuv = iuv.transpose((2, 1, 0))
    i, u, v = iuv[2], iuv[1], iuv[0]

    n_parts = 24
    texture = np.zeros((n_parts, 3, tex_part_size, tex_part_size))

    for part_id in range(1, n_parts + 1):
        im_gen = np.zeros((3, tex_part_size, tex_part_size))

        # try:
        x, y = u[i == part_id], v[i == part_id]
        # transform uv coodrinates to current UV texture coordinates:
        tex_u_coo = (x * (tex_part_size - 1) / 255).astype(int)
        tex_v_coo = (y * (tex_part_size - 1) / 255).astype(int)

        tex_u_coo = np.clip(tex_u_coo, 0, tex_part_size - 1)
        tex_v_coo = np.clip(tex_v_coo, 0, tex_part_size - 1)

        for channel in range(3):
            im_gen[channel][tex_v_coo, tex_u_coo] = im[channel][i == part_id]

        if np.sum(im_gen) > 0:
            im_gen = interpolate_tex(im_gen)
        texture[part_id - 1] = im_gen[:, ::-1, :]
        # except IndexError:
        #     texture[part_id - 1] = im_gen

    tex_trans = np.zeros((24, tex_part_size, tex_part_size, 3))

    for i in range(texture.shape[0]):
        tex_trans[i] = texture[i].transpose(2, 1, 0)

    tex = concat_atlas_tex(tex_trans)
    plt.imshow(tex)
    plt.show()


def parse_iuv(result):
    i = result['pred_densepose'][0].labels.cpu().numpy().astype(float)
    uv = (result['pred_densepose'][0].uv.cpu().numpy() * 255.0).astype(float)
    iuv = np.stack((uv[1, :, :], uv[0, :, :], i))
    iuv = np.transpose(iuv, (1, 2, 0))
    return iuv


def parse_bbox(result):
    return result["pred_boxes_XYXY"][0].cpu().numpy()


def plot_iuv(iuv, bbox):
    image = read_image(IMAGE_FILE, format="BGR")
    image_w, image_h = image.shape[1], image.shape[0]

    bbox[2] = bbox[2] - bbox[0]
    bbox[3] = bbox[3] - bbox[1]
    x, y, w, h = [int(v) for v in bbox]
    bg = np.zeros((image_h, image_w, 3))
    bg[y:y + h, x:x + w, :] = iuv

    plt.imshow(bg / 255)
    plt.show()


def main():
    cfg = setup_config()
    predictor = DefaultPredictor(cfg)
    img = read_image(IMAGE_FILE, format="BGR")  # predictor expects BGR image.
    with torch.no_grad():
        outputs = predictor(img)["instances"]
    results = execute_on_outputs(outputs)
    iuv = parse_iuv(results)
    bbox = parse_bbox(results)
    image = cv2.imread(IMAGE_FILE)[:, :, ::-1]

    plot_iuv(copy.deepcopy(iuv), copy.deepcopy(bbox))
    get_texture(image, iuv, bbox)


if __name__ == '__main__':
    main()
