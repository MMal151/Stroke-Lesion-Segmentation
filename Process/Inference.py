import nibabel as nib
import numpy as np
import os
import pandas as pd
from Process.ProcessUtils import get_metrics_inference, load_model, normalize_img, save_img, thresholding
from Utils.CommonUtils import is_valid_file, str_to_list, str_to_tuple

CLASS_NAME = "[Process/Inference]"


def merge_patches(predicts, in_shape, strategy='max'):
    prediction_lbl = np.zeros(in_shape)

    # For overlapping coordinates, the maximum probability will be used.
    if strategy == 'max':
        for p in predicts:
            curr_patch = p[0]
            (i, j, k) = p[1]
            patch_shape = curr_patch.shape
            prediction_lbl[i:i + patch_shape[0], j:j + patch_shape[1], k:k + patch_shape[2]] = \
                np.maximum(prediction_lbl[i:i + patch_shape[0], j:j + patch_shape[1], k:k + patch_shape[2]], curr_patch)

    return prediction_lbl


def inference(cfg):
    lgr = CLASS_NAME + "[inference()]"

    # cfg["train"] -> Training Configurations
    # Required to ensure the same custom_objects (performance metrics, loss, activations etc are used for testing.)
    model = load_model(cfg["train"], cfg["inference"]["model_path"])

    # Saving the state of some parameters that should match the training configuration.
    normalize = cfg["train"]["data"]["normalize"]
    patch_shape = str_to_tuple(cfg["train"]["data"]["input_shape"])
    perf_metircs = cfg["train"]["perf_metrics"]

    # Overriding the variable for easier usage.
    cfg = cfg["inference"]

    assert is_valid_file(cfg["data_path"]), f"{lgr}: Invalid path for CSV File."
    df_test = pd.read_csv(cfg["data_path"])

    for idx, row in df_test.iterrows():
        img_hdr = nib.load(row['X'])
        img = img_hdr.get_fdata()

        if normalize:
            img = normalize_img(img)

        predicts = []

        for p in str_to_list(row["patches"]):
            (i, j, k) = p # p -> patch coordinates. Expected format: (0, 0, 0)
            # Generating image patch and reshaping it to match the size expected by the model i.e. (batch_size, H, W, D, channel)
            patch = img[i: i + patch_shape[0], j: j + patch_shape[1], k: k + patch_shape[2]].reshape(
                (1, *patch_shape, 1))
            pred = model.predict(patch, batch_size=1)
            predicts.append((pred.reshape(patch_shape), (i, j, k)))

        predicted_lbl = thresholding(merge_patches(predicts, img.shape), cfg["threshold"])

        if row['Y'] is not None or row['Y'] != "":
            org_lbl = nib.load(row['Y']).get_fdata()

            results = get_metrics_inference(org_lbl, predicted_lbl, perf_metircs)

            print(f"{lgr}: Performance Metrics for {row['X']} : {results}")

        if cfg["save_inference"]:
            save_img(predicted_lbl, os.path.join("Test_Results", row['X'].split('/')[-1].split('.nii.gz')[0] + "_Pred.nii.gz"), img_hdr)





