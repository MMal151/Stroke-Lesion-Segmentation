import logging
import os

import nibabel as nib
import random

import numpy as np
from Process.Util import load_model, load_data, save_img
from Util.Metrics import dice_coef
from Util.Postprocessing import thresholding
from Util.Utils import is_valid_file, str_to_tuple, is_valid_dir

CLASS_NAME = "[Process/Test]"


def test(cfg):
    lgr = CLASS_NAME + "[test()]"

    assert is_valid_file(cfg["test"]["model"]["load_path"]), f"{lgr}: Invalid Model Filename/Path. " \
                                                             f"Valid Model is required to perform testing."

    model = load_model(cfg, False)
    input_paths = cfg["test"]["data"]["inputs"].split(",")
    x_test, y_test = [], []
    image_shape = str_to_tuple(cfg["test"]["data"]["image_shape"])

    for i in input_paths:
        x, y = load_data(i, cfg["test"]["data"]["img_ext"], cfg["test"]["data"]["lbl_ext"])
        x_test += x
        y_test += y

    logging.debug(f"{lgr}: Loaded x_test: [{x_test}] \n Loaded y_test: [{y_test}]")

    save_idx = None
    dc_total = 0

    auto_gen_patch = cfg["test"]["data"]["patch"]["stride"] <= 0

    if cfg["test"]["samples"]["save_samples"]:
        logging.info(f"{lgr}: Since, save samples is enabled. Number of samples to be saved: ["
                     + str(cfg["test"]["samples"]["no_samples"]) + "]")
        save_idx = [i for i in range(0, cfg["test"]["samples"]["no_samples"])]

        if cfg["test"]["samples"]["random_samples"]:
            save_idx = random.sample(range(0, len(x_test)), cfg["test"]["samples"]["no_samples"])

        if not is_valid_dir("Test_Results"):
            logging.debug(f"{lgr}: Making directory to save testing results.")
            os.mkdir("Test_Results/")

    for i, (x, y) in enumerate(zip(x_test, y_test)):
        img = nib.load(x).get_fdata()
        lbl = nib.load(y).get_fdata()
        predict_lbl = np.zeros(lbl.shape)

        if cfg["test"]["data"]["patch"]["alw_patching"]:
            patch_shape = str_to_tuple(cfg["test"]["data"]["patch"]["patch_size"])
            if auto_gen_patch:
                patches = generate_patches(img, patch_shape)
            else:
                patches = generate_overlapped_patches(img, patch_shape, cfg["test"]["data"]["patch"]["stride"])
            predicts = []
            logging.debug(
                f"{lgr}: Since, patching is enabled, inference will be done using sliding window technique."
                f"Following patches were generated patches: [{patches}]")
            for p in patches:
                curr_img = p[0].reshape((1, patch_shape[0], patch_shape[1], patch_shape[2], 1))
                pred = model.predict(curr_img)
                predicts.append((pred.reshape(patch_shape), p[1]))
            predict_lbl = merge_patches(predicts, img.shape)
        else:
            predict_lbl = model.predict(img.reshape((1, image_shape[0], image_shape[1], image_shape[2], 1)))

        pred = predict_lbl.reshape(lbl.shape)
        dc = dice_coef(lbl, thresholding(pred))
        logging.info(f"{lgr}: Predicted Dice Coeff for {x}: {dc}")

        dc_total += dc

        if save_idx is not None and i in save_idx:
            save_img(pred, os.path.join("Test_Results", y.split('/')[-1].split('.nii.gz')[0] + "_Pred.nii.gz"))

    if dc_total > 0:
        logging.info(f"{lgr}: Average Dice Coefficient: {dc_total / len(x_test)}")


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
        index = [index[i] + diff[i] for i in range(0, 3)]

    return patches


def generate_overlapped_patches(img, patch_size, stride):
    lgr = CLASS_NAME + "[generate_patches()]"
    if stride < 2:
        logging.info(f"{lgr}: Since num_patches < 2, auto generating patches.")
        return generate_patches(img, stride)
    else:
        patches = []
        diff = [img.shape[i] - patch_size[i] - 1 for i in range(0, 3)]

        assert all(diff[i] > 0 for i in range(0, 3)), f"Patch's size should be smaller than Image's shape. " \
                                                      f"Image Shape: [{img.shape}] < Patch Size: [{patch_size}]"

        for i in range(0, img.shape[0], stride):
            for j in range(0, img.shape[1], stride):
                for k in range(0, img.shape[2], stride):
                    if i + patch_size[0] < img.shape[0] and j + patch_size[1] < img.shape[1] and k + patch_size[2] < \
                            img.shape[2]:
                        patch = img[i: i + patch_size[0], j: j + patch_size[1],
                                k: k + patch_size[2]]
                        patches.append((patch, (i, j, k)))

        return patches


def merge_patches(predicts, org_shape):
    predict_lbl = np.zeros(org_shape)
    pre_idx = None
    for p in predicts:
        curr_patch = p[0]
        (i, j, k) = p[1]
        patch_shape = curr_patch.shape
        predict_lbl[i:i + patch_shape[0], j:j + patch_shape[1], k:k + patch_shape[2]] = \
            predict_lbl[i:i + patch_shape[0], j:j + patch_shape[1], k:k + patch_shape[2]] + curr_patch

        if pre_idx is not None and all(p[1][idx] < pre_idx[idx] + patch_shape[idx] for idx in range(0, 3)):
            (pre_i, pre_j, pre_k) = pre_idx
            predict_lbl[i:pre_i + patch_shape[0], j:pre_j + patch_shape[1], k:pre_k + patch_shape[2]] /= 2
        pre_idx = p[1]

    return predict_lbl
