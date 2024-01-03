import logging
import os.path
import numpy as np
import nibabel as nib
import SimpleITK as sitk

from Misc.CraveMix import generate_new_sample
from Util.Utils import get_random_index, get_all_possible_subdirs, remove_dirs

CLASS_NAME = "[Util/Preprocessing]"


def data_augmentation(cfg, train_x, train_y, cur_input_path):
    lgr = CLASS_NAME + "[data_augmentation()]"
    logging.info(f"{lgr}: Starting data augmentation using factor: " + str(cfg["augmentation"]["factor"])
                 + " and technique: " + cfg["augmentation"]["technique"])

    if cfg["augmentation"]["rem_pre_aug"]:
        logging.info(f"{lgr}: Removing previous augmentations.")
        _ = remove_dirs(get_all_possible_subdirs(cur_input_path, "full_path"), "_cm")

    elif cfg["data"]["apply_augmentation"] and not cfg["augmentation"]["rem_pre_aug"]:
        logging.warning(f"{lgr}: Augmentation is enabled however removing previous augmented datapoints is enabled."
                        f"This might cause discrepancy in data. It is suggested to remove previous datapoints before"
                        f"augmenting new ones.")

    if cfg["augmentation"]["technique"] == "cravemix":
        return augmentation_cm(cfg, train_x, train_y)


def augmentation_cm(cfg, x, y):
    lgr = CLASS_NAME + "[augmentation_cm()]"

    total_aug_dp = int(np.ceil(len(x) * cfg["augmentation"]["factor"]))  # Total number of datapoints to be augmented.
    # Storing the state of len before adding augmented datapoints.
    # This ensures that no augmented datapoint will be used for further augmentation.
    datapoints = len(x)

    if total_aug_dp < datapoints:
        logging.info(f"{lgr}: CraveMix-based Augmentation started. ")

        for k in range(0, total_aug_dp):
            i, j = get_random_index(0, datapoints)
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
            path_new = os.path.join(cfg["data"]["input_path"], new_file)
            os.makedirs(path_new, exist_ok=True)

            img_path = os.path.join(path_new, new_file + "_" + cfg["data"]["img_ext"])
            lbl_path = os.path.join(path_new, new_file + "_" + cfg["data"]["lbl_ext"])

            sitk.WriteImage(new_img, img_path)
            sitk.WriteImage(new_label, lbl_path)
            x.append(img_path)
            y.append(lbl_path)

        logging.info(f"{lgr}: CraveMix-based Augmentation completed. ")

    else:
        logging.error(f"{lgr}: Invalid augmentation factor. "
                      f"Total number of augmented datapoints should not exceed the total number of datapoints.")

    return x, y


# Source: https://github.com/fitushar/3D-Medical-Imaging-Preprocessing-All-you-need
def normalize_img(image):
    img = image.astype(np.float32)

    mean = np.mean(img)
    std = np.std(img)

    if std > 0:
        ret = (img - mean) / std
    else:
        ret = img * 0.
    return ret
