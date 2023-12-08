import logging
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from keras_unet_collection.losses import dice_coef
from tensorflow.keras.callbacks import ModelCheckpoint

from DataGenerators.Nifti3DGenerator import Nifti3DGenerator
from Model.Unet3D import Unet3D
from Util.Utils import get_all_possible_files_paths

CLASS_NAME = "[Process/Train]"


def load_data(input_path, img_ext, lbl_ext):
    lgr = CLASS_NAME + "[load_data()]"

    logging.info(f"{lgr}: Loading Dataset.")

    # Loading dataset
    images = get_all_possible_files_paths(input_path, img_ext)
    labels = get_all_possible_files_paths(input_path, lbl_ext)

    logging.debug(f"{lgr}: Loaded Images: {images} \n Loaded Labels: {labels}")
    return images, labels


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

    x_train, y_train = load_data(cfg["data"]["input_path"], cfg["data"]["img_ext"],
                                 cfg["data"]["lbl_ext"])
    x_valid, y_valid = None, None  # Initializing Validation Set

    if cfg["train"]["valid_ratio"] > 0:
        x_train, x_valid, y_train, y_valid = train_valid_div(x_train, y_train, cfg["train"]["valid_ratio"],
                                                             cfg["data"]["seed"])

    logging.info(f"{lgr}: Creating Generators for training (& validation) data.")
    train_gen = Nifti3DGenerator(cfg, x_train, y_train)
    valid_gen = None
    if x_valid is not None:
        valid_gen = Nifti3DGenerator(cfg, x_valid, y_valid)

    logging.info(f"{lgr}: Generating Model.")

    monitor = 'loss'
    if valid_gen is not None:
        monitor = "val_loss"

    if strategy is not None:
        with strategy.scope():
            model = Unet3D(cfg).generate_model()
            # TO-DO: 1. Need to make optimizer configurable. 2. Implement learning rate schedular. 3. Make loss and metrics configurable.
            model.compile(optimizer=Adam(learning_rate=cfg["train"]["learning_rate"]), loss='binary_crossentropy',
                          metrics=[dice_coef])

            checkpoint = ModelCheckpoint(cfg["train"]["model_name"] + "{epoch:02d}.h5", monitor=monitor,
                                         save_best_only=True,
                                         save_freq='epoch')
            history = model.fit(train_gen, validation_data=valid_gen, steps_per_epoch=len(train_gen),
                                epochs=cfg["train"]["epochs"], callbacks=[checkpoint])
    else:
        model = Unet3D(cfg).generate_model()
        # TO-DO: 1. Need to make optimizer configurable. 2. Implement learning rate schedular. 3. Make loss and metrics configurable.
        model.compile(optimizer=Adam(learning_rate=cfg["train"]["learning_rate"]), loss='binary_crossentropy',
                      metrics=[dice_coef])
        checkpoint = ModelCheckpoint(cfg["train"]["model_name"] + "{epoch:02d}.h5", monitor=monitor,
                                     save_best_only=True,
                                     save_freq='epoch')
        history = model.fit(train_gen, validation_data=valid_gen, steps_per_epoch=len(train_gen),
                            epochs=cfg["train"]["epochs"], callbacks=[checkpoint])
