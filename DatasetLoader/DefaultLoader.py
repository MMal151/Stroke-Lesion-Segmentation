import logging

from sklearn.model_selection import train_test_split

from Process.Utils import load_data
from Util.Preprocessing import data_augmentation
from Util.Utils import remove_dirs, get_all_possible_subdirs, get_all_possible_files_paths

CLASS_NAME = "[DatasetLoader/DefaultLoader]"


def load_dataset(input_path, img_ext, lbl_ext):
    lgr = CLASS_NAME + "[load_data()]"

    logging.info(f"{lgr}: Loading Dataset.")

    # Loading dataset
    images = get_all_possible_files_paths(input_path, img_ext)
    labels = get_all_possible_files_paths(input_path, lbl_ext)

    logging.debug(f"{lgr}: Loaded Images: {images} \n Loaded Labels: {labels}")
    return images, labels


# Divides dataset into training and validation set
def train_valid_div(images, labels, valid_ratio, seed=2023, div_type='Validation'):
    lgr = CLASS_NAME + "[train_valid_div()]"
    logging.debug(f"{lgr}: Starting train/valid division.")

    x_train, x_valid, y_train, y_valid = train_test_split(images, labels, test_size=valid_ratio,
                                                          random_state=seed)

    logging.debug(f"{lgr}: x_train: {x_train} \n y_train: {y_train} \n x_{div_type}: {x_valid} \n "
                  f"y_{div_type}: {y_valid}")
    logging.info(f"{lgr}: Training-Instances: {len(x_train)}. "
                 f"{div_type}-Instances: {len(x_valid)}")

    return x_train, x_valid, y_train, y_valid


def loader(cfg):
    lgr = CLASS_NAME + "[loader()]"

    input_paths = cfg["train"]["data"]["inputs"].split(",")
    x_train, y_train, x_valid, y_valid, x_test, y_test = [], [], [], [], [], []  # Initializing Validation Set

    for i in input_paths:
        if cfg["train"]["data"]["rem_pre_aug"]:
            logging.info(f"{lgr}: Removing previous augmentations.")
            _ = remove_dirs(get_all_possible_subdirs(i, "full_path"), "_cm")

        x, y = load_data(i.strip(), cfg["train"]["data"]["img_ext"], cfg["train"]["data"]["lbl_ext"])

        if cfg["train"]["data"]["test"]["alw_test"] and cfg["train"]["data"]["test"]["ratio"] > 0:
            logging.info(f"{lgr}: Separating test data from training and validation set for data source {i}")
            x, x_temp, y, y_temp = train_valid_div(x, y, cfg["train"]["data"]["test"]["ratio"],
                                                   cfg["train"]["data"]["test"]["seed"], 'Test')
            x_test = x_temp + x_test
            y_test = y_temp + y_test
            logging.debug(f"{lgr}: State after merging with test sets. x_test = {x_test} \n y_test = {y_test}")

        if cfg["train"]["data"]["valid"]["ratio"] > 0:
            logging.info(f"{lgr}: Separating validation data from training and validation set for data source {i}")
            x, x_val, y, y_val = train_valid_div(x, y, cfg["train"]["data"]["valid"]["ratio"],
                                                 cfg["train"]["data"]["valid"]["seed"])
            x_valid = x_valid + x_val
            y_valid = y_valid + y_val
            logging.debug(
                f"{lgr}: State after merging with validation sets. x_valid = {x_valid} \n y_valid = {y_valid}")

        if cfg["train"]["data"]["augmentation"]["alw_aug"]:
            logging.info(f"{lgr}: Applying augmentation to training data for data source {i} ")
            x, y = data_augmentation(cfg, x, y, i)

        logging.debug(f"{lgr}: State before merging with test sets. x = {x} \n y = {y} \n "
                      f"x_train = {x_train} \n y_train = {y_train}")
        x_train = x_train + x
        y_train = y_train + y
        logging.debug(f"{lgr}: State before merging with test sets. x_train = {x_train} \n y_train = {y_train}")

    return x_train, y_train, x_valid, y_valid, x_test, y_test
