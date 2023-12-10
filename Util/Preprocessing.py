import logging
import random

import numpy as np

from Util.Utils import get_random_index

CLASS_NAME = "[Util/Preprocessing]"


def data_augmentation(cfg, train_x, train_y):
    lgr = CLASS_NAME + "[data_augmentation()]"
    logging.info(f"{lgr}: Starting data augmentation using factor: " + str(cfg["augmentation"]["factor"])
                 + " and technique: " + cfg["augmentation"]["technique"])

    if cfg["augmentation"]["technique"] == "cravemix":
        return augmentation_cm(train_x, train_y, cfg["augmentation"]["factor"])


def augmentation_cm(x, y, factor):
    lgr = CLASS_NAME + "[augmentation_cm()]"
    print("UNDER CONSTRUCTION")

    total_aug_dp = int(np.ceil(len(x) * factor))  # Total number of datapoints to be augmented.
    augmented_samples = []

    if total_aug_dp < len(x):
        for k in range(0, total_aug_dp):
            i, j = get_random_index(0, len(x))
            logging.debug(f"{lgr}: Augmenting Datapoints {x[i]} and {x[j]}")
    else:
        logging.error(f"{lgr}: Invalid augmentation factor. "
                      f"Total number of augmented datapoints should not exceed the total number of datapoints.")

    return x, y
