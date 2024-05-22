import logging
import os.path
import numpy as np
import nibabel as nib
import SimpleITK as sitk

from Misc.CarveMix import generate_new_sample
from Util.Utils import get_random_index

CLASS_NAME = "[Util/Preprocessing]"


def data_augmentation(cfg, train_x, train_y, in_path):
    lgr = CLASS_NAME + "[data_augmentation()]"
    logging.info(f"{lgr}: Starting data augmentation using factor: "
                 + str(cfg["train"]["data"]["augmentation"]["factor"])
                 + " and technique: " + cfg["train"]["data"]["augmentation"]["technique"])

    if cfg["train"]["data"]["augmentation"]["alw_aug"] and not cfg["train"]["data"]["rem_pre_aug"]:
        logging.warning(f"{lgr}: Augmentation is enabled however removing previous augmented datapoints is enabled."
                        f"This might cause discrepancy in data. It is suggested to remove previous datapoints before"
                        f"augmenting new ones.")

    if cfg["train"]["data"]["augmentation"]["technique"] == "cravemix":
        return augmentation_cm(cfg, train_x, train_y, in_path)


def augmentation_cm(cfg, x, y, in_path):
    lgr = CLASS_NAME + "[augmentation_cm()]"

    # Total number of datapoints to be augmented.
    total_aug_dp = int(np.floor(len(x) * cfg["train"]["data"]["augmentation"]["factor"]))

    # Storing the state of len before adding augmented datapoints.
    # This ensures that no augmented datapoint will be used for further augmentation.
    datapoints = len(x)

    if total_aug_dp < datapoints:
        logging.info(f"{lgr}: CraveMix-based Augmentation started. ")

        for k in range(0, total_aug_dp):
            i, j = get_random_index(0, datapoints - 1)
            logging.debug(f"{lgr}: Augmenting Datapoints {x[i]} and {x[j]}")
            vol_a, vol_b = nib.imagestats.mask_volume(nib.load(y[i])), nib.imagestats.mask_volume(nib.load(y[j]))

            # This comparison is missing from the original cravemix algorithm. Having this comparison however ensures
            # smoother augmentation. The image given as input first acts as the base image. If the bigger lesion is
            # augmented in the image imbalances the intensities of the base image.
            if vol_a > vol_b:
                new_img, new_label, _, _ = generate_new_sample(x[i], x[j], y[i], y[j])
            else:
                new_img, new_label, _, _ = generate_new_sample(x[j], x[i], y[j], y[i])

            new_file = x[i].split("/")[-2] + "_" + x[j].split("/")[-2] + "_cm"
            path_new = os.path.join(in_path, new_file)
            os.makedirs(path_new, exist_ok=True)

            img_path = os.path.join(path_new, new_file + "_" + cfg["train"]["data"]["img_ext"])
            lbl_path = os.path.join(path_new, new_file + "_" + cfg["train"]["data"]["lbl_ext"])

            sitk.WriteImage(new_img, img_path)
            sitk.WriteImage(new_label, lbl_path)
            x.append(img_path)
            y.append(lbl_path)

        logging.info(f"{lgr}: CarveMix-based Augmentation completed. ")

    else:
        logging.error(f"{lgr}: Invalid augmentation factor. "
                      f"Total number of augmented datapoints should not exceed the total number of datapoints.")

    return x, y


# Source: https://github.com/fitushar/3D-Medical-Imaging-Preprocessing-All-you-need
def normalize_img(image, smooth=1e-8):
    mean = np.mean(image)
    std = np.std(image)
    image -= mean
    image /= (max(std, smooth))
    return image


def random_patch_3D(img, lbl, target_shape):
    assert img.shape == lbl.shape, f"Image and Label should be of the same shape. " \
                                   f"Image's Shape: [{img.shape}] != Label's Shape: [{lbl.shape}]"
    dif = [img.shape[i] - target_shape[i] - 1 for i in range(0, 3)]

    assert all(dif[i] > 0 for i in range(0, 3)), f"Patch's size should be smaller than Image's shape. " \
                                                 f"Image Shape: [{img.shape}] < Patch Size: [{target_shape}]"

    i, j, k = np.random.randint((0, 0, 0), tuple(dif))

    img_patch = img[i:i + target_shape[0], j:j + target_shape[1], k:k + target_shape[2]]
    lbl_patch = lbl[i:i + target_shape[0], j:j + target_shape[1], k:k + target_shape[2]]

    return img_patch, lbl_patch
