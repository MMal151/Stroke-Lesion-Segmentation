import logging

import nibabel as nib
import numpy as np
import tensorflow as tf
from keras_unet_collection.activations import GELU, Snake

from Util.Loss import CUSTOM_LOSS_FUNCTIONS, get_loss, dice
from Util.Metrics import get_metrics_test
from Util.Utils import get_all_possible_files_paths, is_valid_file

CLASS_NAME = "[Process/Utilities]"
CUSTOM_ACTIVATIONS = "gelu,snake"


def load_data(input_path, img_ext, lbl_ext):
    lgr = CLASS_NAME + "[load_data()]"

    logging.info(f"{lgr}: Loading Dataset.")

    # Loading dataset
    images = get_all_possible_files_paths(input_path, img_ext)
    labels = get_all_possible_files_paths(input_path, lbl_ext)

    logging.debug(f"{lgr}: Loaded Images: {images} \n Loaded Labels: {labels}")
    return images, labels


# Custom Objects includes all custom activations, loss functions, and performance metrics.
def get_custom_objects(cfg):
    lgr = CLASS_NAME + "[get_custom_objects()]"
    custom_objects = get_metrics_test(cfg["train"]["perf_metrics"])
    if CUSTOM_ACTIVATIONS.__contains__(cfg["train"]["activation"].lower()):
        logging.debug(f"{lgr}: Adding custom activation function in custom_obj list.")
        if cfg["train"]["activation"].lower() == "gelu":
            custom_objects["GELU"] = GELU()
        elif cfg["train"]["activation"].lower() == "snake":
            custom_objects["Snake"] = Snake()

    if CUSTOM_LOSS_FUNCTIONS.__contains__(cfg["train"]["loss"].lower()):
        logging.debug(f"{lgr}: Adding custom loss function in custom_obj list.")
        loss = get_loss(cfg)
        custom_objects[cfg["train"]["loss"]] = loss

        #if cfg["train"]["loss"].lower() == "focal_tversky":
         #   custom_objects["focal_tversky"] = loss

    return custom_objects


def load_model(cfg, resume_train):
    model = None
    if resume_train:
        model_file = cfg["train"]["resume"]["model_path"]
    else:
        model_file = cfg["test"]["model"]["load_path"]

    if is_valid_file(model_file):
        model = tf.keras.models.load_model(model_file,
                                           custom_objects=get_custom_objects(cfg))

    return model


def print_train_configurations(cfg):
    lgr = CLASS_NAME + "[print_train_configurations()]"

    logging.info(f"{lgr}: Following configurations were loaded from configuration file.")

    logging.info(f"\n #------------- General Configurations -------------#")

    logging.info(f"\n #------------ GPU-Based Configurations ------------#")
    logging.info(f"Allow parallel processing of data points across multiple GPUs: ["
                 + str(cfg["misc"]["gpu"]["alw_para_prs"]) + "]")
    logging.info(f"Parallel processing strategy: [" + cfg["misc"]["gpu"]["strategy"] + "]")
    logging.info(f"GPUs to be used for processing: [" + cfg["misc"]["gpu"]["no_gpus"] + "]")

    logging.info(f"\n #------------ Training Configurations -------------#")
    logging.info(f"Model type: [" + cfg["train"]["model_type"] + "]")
    logging.info(f"Total number of iterations: [" + str(cfg["train"]["num_iter"]) + "]")
    logging.info(f"Epochs: [" + str(cfg["train"]["epochs"]) + "]")

    logging.info(f"Performance Metrics to be used: ")
    if cfg["train"]["perf_metrics"].__contains__("acc"):
        logging.info(f"\n Accuracy")
    if cfg["train"]["perf_metrics"].__contains__("mean_iou"):
        logging.info(f"\n MeanIoU")
    if cfg["train"]["perf_metrics"].__contains__("recall"):
        logging.info(f"\n Recall")
    if cfg["train"]["perf_metrics"].__contains__("prec"):
        logging.info(f"\n Precision")
    if cfg["train"]["perf_metrics"].__contains__("dice_coef"):
        logging.info(f"\n Dice Coefficient")

    logging.info(f"Activation Function to be used: [" + cfg["train"]["activation"] + "]")
    logging.info(f"Output Classes: [" + str(cfg["train"]["output_classes"]) + "]")
    logging.info(f"Dropout: [" + str(cfg["train"]["dropout"]) + "]")
    logging.info(f"Minimum Filter Size: [" + str(cfg["train"]["min_filter"]) + "]")
    logging.info(f"Optimizer: [" + cfg["train"]["optimizer"] + "]")
    logging.info(f"Learning Rate: [" + str(cfg["train"]["learning_rate"]) + "]")
    logging.info(f"Apply Learning Rate Schedular: [" + str(cfg["train"]["aply_lr_sch"]) + "]")
    logging.info(f"Loss Function: [" + str(cfg["train"]["loss"]) + "]")
    logging.info(f"Model Save Path / Filename: [" + str(cfg["train"]["save"]["model_name"]) + "]")
    logging.info(f"Save best model only: [" + str(cfg["train"]["save"]["best_only"]) + "]")

    logging.info(f"\n #--------- Training Data Configurations -----------#")
    logging.info(f"Input Sources: [" + cfg["train"]["data"]["inputs"] + "]")
    logging.info(f"Input Scan;s Extension: [" + cfg["train"]["data"]["img_ext"] + "]")
    logging.info(f"Label's Extension: [" + cfg["train"]["data"]["lbl_ext"] + "]")
    logging.info(f"Image Shape: [" + cfg["train"]["data"]["image_shape"] + "]")
    logging.info(f"Batch Size: [" + str(cfg["train"]["data"]["batch_size"]) + "]")
    logging.info(f"Normalize Data: [" + str(cfg["train"]["data"]["norm_data"]) + "]")
    logging.info(f"Shuffle Data on each epoch: [" + str(cfg["train"]["data"]["shuffle"]) + "]")
    logging.info(f"Remove previously augmented datapoints: [" + str(cfg["train"]["data"]["rem_pre_aug"]) + "]")
    logging.info(f"Training/Validation Division Seed: [" + str(cfg["train"]["data"]["valid"]["seed"]) + "]")
    logging.info(f"Validation Ratio: [" + str(cfg["train"]["data"]["valid"]["ratio"]) + "]")

    if cfg["train"]["data"]["test"]["alw_test"]:
        logging.info(f"\n #---------- Testing Data Configurations -----------#")
        logging.info(f"Testing Ratio: [" + str(cfg["train"]["data"]["test"]["ratio"]) + "]")
        logging.info(f"Division Seed: [" + str(cfg["train"]["data"]["test"]["seed"]) + "]")

    if cfg["train"]["data"]["augmentation"]["alw_aug"]:
        logging.info(f"\n #------- Data Augmentation Configurations ---------#")
        logging.info(f"Augmentation Factor: [" + str(cfg["train"]["data"]["augmentation"]["factor"]) + "]")
        logging.info(f"Augmentation Technique: [" +
                     cfg["train"]["data"]["augmentation"]["technique"] + "]")

    if cfg["train"]["data"]["patch"]["alw_patching"]:
        logging.info(f"\n #--------- Patch Related Configurations -----------#")
        logging.info(f"Patch Shape: [" + cfg["train"]["data"]["patch"]["patch_size"] + "]")
        logging.info(f"Total Patches: [" + str(cfg["train"]["data"]["patch"]["total_patches"]) + "]")
        logging.info(f"Random Patches: [" + str(cfg["train"]["data"]["patch"]["random_patches"]) + "]")

    if cfg["train"]["resume"]["resume_train"]:
        logging.info(f"Model Path: [" + cfg["train"]["resume"]["model_path"] + "]")


def print_test_configurations(cfg):
    lgr = CLASS_NAME + "[print_test_configurations()]"
    logging.info(f"{lgr}: Following configurations were loaded from configuration file.")

    logging.info(f"\n #------------- Testing Configurations -------------#")
    logging.info(f"Model Load Path: [" + cfg["test"]["model"]["load_path"] + "]")

    logging.info(f"\n #---------- Testing Data Configurations -----------#")
    logging.info(f"Input Sources: [" + cfg["test"]["data"]["inputs"] + "]")
    logging.info(f"Input Scan;s Extension: [" + cfg["test"]["data"]["img_ext"] + "]")
    logging.info(f"Label's Extension: [" + cfg["test"]["data"]["lbl_ext"] + "]")
    logging.info(f"Image Shape: [" + cfg["test"]["data"]["image_shape"] + "]")

    if cfg["test"]["data"]["patch"]["alw_patching"]:
        logging.info(f"\n #--------- Patch Related Configurations -----------#")
        logging.info(f"Patch Shape: [" + cfg["test"]["data"]["patch"]["patch_size"] + "]")
        logging.info(f"Stride: [" + str(cfg["test"]["data"]["patch"]["stride"]) + "]")

    if cfg["test"]["samples"]["save_samples"]:
        logging.info(f"\n # Save Random Samples for Visualization of Results #")
        logging.info(f"Number of samples to save: [" + str(cfg["test"]["samples"]["no_samples"]) + "]")
        logging.info(f"Save random samples: [" + str(cfg["test"]["samples"]["random_samples"]) + "]")


def save_img(img, filename):
    ni = nib.Nifti1Image(img, affine=np.eye(4))
    nib.save(ni, filename)


def generate_patch_idx(image_shape, stride, patch_shape, repeat=1):
    lgr = CLASS_NAME + "[generate_patch_idx()]"
    idx = []
    total_patches = 0
    logging.debug(
        f"{lgr}: Input values: image_shape: [{image_shape}], stride: [{stride}], patch_shape: [{patch_shape}].")

    assert all(image_shape[i] > 0 and patch_shape[i] > 0 for i in range(0, 3)), \
        f"{lgr}: Invalid image shape [{image_shape}] or patch_shape [{patch_shape}]. All values should be " \
        f"greater than zero."
    assert repeat >= 1, f"{lgr}: Invalid value of for repeat [{repeat}]."

    for i in range(0, image_shape[0], stride):
        for j in range(0, image_shape[1], stride):
            for k in range(0, image_shape[2], stride):
                if (i + patch_shape[0]) < image_shape[0] and (j + patch_shape[1]) < image_shape[1] and \
                        (k + patch_shape[2] < image_shape[2]):
                    total_patches += 1
                    for m in range(0, repeat):
                        idx.append((i, j, k))

    logging.debug(f"{lgr}: Generated Indexes: {idx}")
    return total_patches, idx
