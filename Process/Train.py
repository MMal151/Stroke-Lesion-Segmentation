import logging
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from keras_unet_collection.losses import dice_coef
from tensorflow.keras.callbacks import ModelCheckpoint

from DataGenerators.Nifti3DGenerator import Nifti3DGenerator
from Model.Unet3D import Unet3D
from Model.Vnet import Vnet
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

    input_paths = cfg["data"]["input_path"].split(",")
    x_train, y_train, x_valid, y_valid, x_test, y_test = [], [], [], [], [], []  # Initializing Validation Set

    for i in input_paths:
        x, y = load_data(i.strip(), cfg["data"]["img_ext"], cfg["data"]["lbl_ext"])

        if cfg["train"]["test_on_same_data"] and cfg["train"]["test_ratio"] > 0:
            logging.info(f"{lgr}: Separating test data from training and validation set for data source {i}")
            x, x_temp, y, y_temp = train_valid_div(x, y, cfg["train"]["test_ratio"], cfg["data"]["seed"])
            logging.debug(f"{lgr}: State before merging with test sets. x_temp = {x_temp} \n y_temp = {y_temp} \n "
                          f"x_test = {x_test} \n y_test = {y_test}")
            x_test = x_temp + x_test
            y_test = y_temp + y_test
            logging.debug(f"{lgr}: State after merging with test sets. x_test = {x_test} \n y_test = {y_test}")

        if cfg["train"]["valid_ratio"] > 0:
            logging.info(f"{lgr}: Separating validation data from training and validation set for data source {i}")
            x, x_val, y, y_val = train_valid_div(x, y, cfg["train"]["valid_ratio"], cfg["data"]["seed"])
            logging.debug(f"{lgr}: State before merging with validation sets. x_val = {x_val} \n x_val = {x_val} \n "
                          f"x_valid = {x_valid} \n y_valid = {y_valid}")
            x_valid = x_valid + x_val
            y_valid = y_valid + y_val
            logging.debug(
                f"{lgr}: State before merging with validation sets. x_valid = {x_valid} \n y_valid = {y_valid}")

        if cfg["data"]["apply_augmentation"]:
            logging.info(f"{lgr}: Applying augmentation to training data for data source {i} ")
            x, y = data_augmentation(cfg, x, y, i)

        logging.debug(f"{lgr}: State before merging with test sets. x = {x} \n y = {y} \n "
                      f"x_train = {x_train} \n y_train = {y_train}")
        x_train = x_train + x
        y_train = y_train + y
        logging.debug(f"{lgr}: State before merging with test sets. x_train = {x_train} \n y_train = {y_train}")

    logging.info(f"{lgr}: Creating Generators for training (& validation & test) data.")
    train_gen = Nifti3DGenerator(cfg, x_train, y_train)
    valid_gen, test_gen = None, None
    if x_valid is not []:
        valid_gen = Nifti3DGenerator(cfg, x_valid, y_valid)
    if x_test is not []:
        test_gen = Nifti3DGenerator(cfg, x_test, y_test)

    logging.info(f"{lgr}: Generating Model.")

    if strategy is not None:
        with strategy.scope():
            fit_model(cfg, train_gen, valid_gen, test_gen)
    else:
        fit_model(cfg, train_gen, valid_gen, test_gen)


def fit_model(cfg, train_gen, valid_gen, test_gen):
    lgr = CLASS_NAME + "[fit_model()]"

    model = None
    if cfg["common_config"]["model_type"] == "unet":
        model = Unet3D(cfg).generate_model()
    elif cfg["common_config"]["model_type"] == "vnet":
        model = Vnet(cfg).generate_model()

    if model is not None:
        monitor = 'loss'
        validation = False
        if valid_gen is not None:
            monitor = "val_loss"
            validation = True
        # TO-DO: 1. Need to make optimizer configurable. 2. Implement learning rate schedular. 3. Make loss and metrics configurable.
        model.compile(optimizer=Adam(learning_rate=cfg["train"]["learning_rate"]), loss='binary_crossentropy',
                      metrics=[dice_coef])
        checkpoint = ModelCheckpoint(cfg["train"]["model_name"] + "{epoch:02d}.h5", monitor=monitor,
                                     save_best_only=cfg["train"]["save_best_only"],
                                     save_freq='epoch')
        history = model.fit(train_gen, validation_data=valid_gen, steps_per_epoch=len(train_gen),
                            epochs=cfg["train"]["epochs"], callbacks=[checkpoint])
        show_history(history, validation)

        if test_gen is not None:
            logging.info(f"{lgr}: Starting testing.")
            loss, metric = model.evaluate(test_gen, batch_size=1, steps=test_gen.get_x_len())
            print(f"{lgr}: Testing Loss: {loss} \n Testing Dice-Coeff: {metric}")
    else:
        logging.error(f"{lgr}: Invalid model_type. Aborting training process.")
