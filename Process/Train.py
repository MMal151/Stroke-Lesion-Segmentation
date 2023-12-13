import logging
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from keras_unet_collection.losses import dice_coef
from tensorflow.keras.callbacks import ModelCheckpoint

from DataGenerators.Nifti3DGenerator import Nifti3DGenerator
from Model.Unet3D import Unet3D
from Process.Utilities import load_data
from Util.Preprocessing import data_augmentation
from Util.Utils import get_all_possible_subdirs, remove_dirs
from Util.Visualization import show_history

CLASS_NAME = "[Process/Train]"


# Divides dataset into training and validation set
def train_valid_div(images, labels, valid_ratio, seed=2023):
    lgr = CLASS_NAME + "[train_valid_div()]"
    logging.debug(f"{lgr}: Starting train/valid division.")

    x_train, x_valid, y_train, y_valid = train_test_split(images, labels, test_size=valid_ratio,
                                                          random_state=seed)

    logging.debug(f"{lgr}: x_train: {x_train} \n y_train: {y_train} \n x_valid: {x_valid} \n "
                  f"y_valid: {y_valid}")
    logging.info(f"{lgr}: Training-Instances: {len(x_train)}. "
                 f"Validation-Instances: {len(x_valid)}")

    return x_train, x_valid, y_train, y_valid


def train(cfg, strategy=None):
    lgr = CLASS_NAME + "[train()]"
    logging.info(f"{lgr}: Starting Training.")

    if cfg["data"]["apply_augmentation"] and cfg["augmentation"]["rem_pre_aug"]:
        logging.info(f"{lgr}: Removing previous augmentations.")
        _ = remove_dirs(get_all_possible_subdirs(cfg["data"]["input_path"], "full_path"), "_cm")

    elif cfg["data"]["apply_augmentation"] and not cfg["augmentation"]["rem_pre_aug"]:
        logging.warning(f"{lgr}: Augmentation is enabled however removing previous augmented datapoints is enabled."
                        f"This might cause discrepancy in data. It is suggested to remove previous datapoints before"
                        f"augmenting new ones.")

    x_train, y_train = load_data(cfg["data"]["input_path"], cfg["data"]["img_ext"],
                                 cfg["data"]["lbl_ext"])
    x_valid, y_valid = None, None  # Initializing Validation Set

    if cfg["train"]["valid_ratio"] > 0:
        x_train, x_valid, y_train, y_valid = train_valid_div(x_train, y_train, cfg["train"]["valid_ratio"],
                                                             cfg["data"]["seed"])

    if cfg["data"]["apply_augmentation"]:
        logging.info(f"{lgr}: Applying augmentation to training data. ")
        x_train, y_train = data_augmentation(cfg, x_train, y_train)

    logging.info(f"{lgr}: Creating Generators for training (& validation) data.")
    train_gen = Nifti3DGenerator(cfg, x_train, y_train)
    valid_gen = None
    if x_valid is not None:
        valid_gen = Nifti3DGenerator(cfg, x_valid, y_valid)

    logging.info(f"{lgr}: Generating Model.")

    if strategy is not None:
        with strategy.scope():
            fit_model(cfg, train_gen, valid_gen)
    else:
        fit_model(cfg, train_gen, valid_gen)


def fit_model(cfg, train_gen, valid_gen):
    lgr = CLASS_NAME + "[fit_model()]"

    model = None
    if cfg["common_config"]["model_type"] == "unet":
        model = Unet3D(cfg).generate_model()

    if model is not None:
        monitor = 'loss'
        validation = False
        if valid_gen is not None:
            monitor = "val_dice_coef"
            validation = True
        # TO-DO: 1. Need to make optimizer configurable. 2. Implement learning rate schedular. 3. Make loss and metrics configurable.
        model.compile(optimizer=Adam(learning_rate=cfg["train"]["learning_rate"]), loss='binary_crossentropy',
                      metrics=[dice_coef])
        checkpoint = ModelCheckpoint(cfg["train"]["model_name"] + "{epoch:02d}.h5", monitor=monitor,
                                     save_best_only=True,
                                     save_freq='epoch')
        history = model.fit(train_gen, validation_data=valid_gen, steps_per_epoch=len(train_gen),
                            epochs=cfg["train"]["epochs"], callbacks=[checkpoint])
        show_history(history, validation)
    else:
        logging.error(f"{lgr}: Invalid model_type. Aborting training process.")
