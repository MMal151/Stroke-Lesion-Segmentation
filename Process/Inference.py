import logging
import nibabel as nib
import numpy as np

from Process.Utilities import load_model, load_data
from Util.Metrics import dice_coef
from Util.Utils import is_valid_file, str_to_tuple

CLASS_NAME = "[Process/Inference]"


def get_segmentation(cfg):
    lgr = CLASS_NAME + "[get_segmentation()]"
    if is_valid_file(cfg["test"]["model_load_state"]):
        model = load_model(cfg, False)
        x_test, y_test = load_data(cfg["data"]["input_path"], cfg["data"]["img_ext"],
                                   cfg["data"]["lbl_ext"])
        image_shape = str_to_tuple(cfg["data"]["image_shape"])
        dc_total = 0

        for i, (x, y) in enumerate(zip(x_test, y_test)):
            img = nib.load(x).get_fdata()
            lbl = nib.load(y).get_fdata()

            if cfg["data"]["aply_patch"]:
                patches = generate_patches(img, image_shape)
                predicts = []
                logging.debug(
                    f"{lgr}: Since, patching is enabled, inference will be done using sliding window technique."
                    f"Following patches were generated patches: [{patches}]")
                for p in patches:
                    curr_img = p[0].reshape((1, image_shape[0], image_shape[1], image_shape[2], 1))
                    pred = model.predict(curr_img)
                    predicts.append((pred.reshape(image_shape), p[1]))

                predict_lbl = thresholding(merge_patches(predicts, img.shape))
            else:

                predict_lbl = thresholding(model.predict
                                           (img.reshape((1, image_shape[0], image_shape[1], image_shape[2], 1))))
            pred = predict_lbl.reshape(lbl.shape)
            dc = dice_coef(lbl, pred)

            logging.info(f"{lgr}: Dice-Coef for [{x_test}] is [{dc}]. ")
            dc_total = dc_total + dc

    if dc_total > 0:
        logging.info(f"Average Dice Coefficient: [{dc_total / len(x_test)}]")

    else:
        logging.error(f"{lgr}: Invalid Model Path.")


def generate_patches(img, target_shape):
    patches = []
    org_shape = img.shape
    diff = [img.shape[i] - target_shape[i] - 1 for i in range(0, 3)]

    assert all(diff[i] > 0 for i in range(0, 3)), f"Patch's size should be smaller than Image's shape. " \
                                                  f"Image Shape: [{img.shape}] < Patch Size: [{target_shape}]"

    index = [0, 0, 0]

    while all(index[i] + target_shape[i] < org_shape[i] for i in range(0, 3)):
        patch = img[index[0]:index[0] + target_shape[0],
                    index[1]:index[1] + target_shape[1],
                    index[2]:index[2] + target_shape[2]]
        patches.append((patch, index))
        index = [index[i] + diff[i] for i in range(0,3)]

    return patches


def merge_patches(predicts, org_shape):
    predict_lbl = np.zeros(org_shape)
    for p in predicts:
        curr_patch = p[0]
        (i, j, k) = p[1]
        patch_shape = curr_patch.shape
        predict_lbl[i:i + patch_shape[0], j:j + patch_shape[1], k:k + patch_shape[2]] = curr_patch

    return predict_lbl


def thresholding(lbl, thresh=0.8):
    for i in range(0, lbl.shape[0]):
        for j in range(0, lbl.shape[1]):
            for k in range(0, lbl.shape[2]):
                if lbl[i][j][k] > thresh:
                    lbl[i][j][k] = 1
                else:
                    lbl[i][j][k] = 0
    return lbl
