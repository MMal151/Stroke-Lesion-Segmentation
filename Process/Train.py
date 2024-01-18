import logging
from time import time

from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint

from DataGenerators.Nifti3DGenerator import Nifti3DGenerator
from Model.Unet3D import Unet3D
from Model.Vnet import Vnet
from Process.Test import log_test_results
from Process.Utilities import load_data, load_model
from Util.Loss import get_loss
from Util.Metrics import get_metrics
from Util.Optimizers import get_optimizer
from Util.Preprocessing import data_augmentation
from Util.Utils import get_all_possible_subdirs, remove_dirs
from Util.Visualization import show_history

CLASS_NAME = "[Process/Train]"


# Divides dataset into training and validation set
def train_valid_div(images, labels, valid_ratio, seed=2023, div_type='Validation'):
    lgr = CLASS_NAME + "[train_valid_div()]"
    logging.debug(f"{lgr}: Starting train/valid division.")

    x_train, x_valid, y_train, y_valid = train_test_split(images, labels, test_size=valid_ratio,
                                                          random_state=seed)

    logging.debug(f"{lgr}: x_train: {x_train} \n y_train: {y_train} \n x_valid: {x_valid} \n "
                  f"y_valid: {y_valid}")
    logging.info(f"{lgr}: Training-Instances: {len(x_train)}. "
                 f"{div_type}-Instances: {len(x_valid)}")

    return x_train, x_valid, y_train, y_valid


def train(cfg, strategy=None):
    lgr = CLASS_NAME + "[train()]"
    logging.info(f"{lgr}: Starting Training.")

    input_paths = cfg["data"]["input_path"].split(",")
    x_train, y_train, x_valid, y_valid, x_test, y_test = [], [], [], [], [], []  # Initializing Validation Set

    for i in input_paths:
        if cfg["augmentation"]["rem_pre_aug"]:
            logging.info(f"{lgr}: Removing previous augmentations.")
            _ = remove_dirs(get_all_possible_subdirs(i, "full_path"), "_cm")

        x, y = load_data(i.strip(), cfg["data"]["img_ext"], cfg["data"]["lbl_ext"])

        if cfg["train"]["test_on_same_data"] and cfg["train"]["test_ratio"] > 0:
            logging.info(f"{lgr}: Separating test data from training and validation set for data source {i}")
            x, x_temp, y, y_temp = train_valid_div(x, y, cfg["train"]["test_ratio"], cfg["data"]["seed"], 'Test')
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

    start_time = time()
    model = load_model(cfg, cfg["train"]["resume_training"])

    if cfg["common_config"]["model_type"].lower() == "unet" and model is None:
        model = Unet3D(cfg).generate_model()
    elif cfg["common_config"]["model_type"].lower() == "vnet" and model is None:
        model = Vnet(cfg).generate_model()

    if model is not None:
        monitor = 'loss'
        validation = False
        if valid_gen is not None:
            monitor = "val_loss"
            validation = True

        metrics, eval_list = get_metrics(cfg["train"]["perf_metrics"])
        # TO-DO: 1. Need to make optimizer configurable. 2. Implement learning rate schedular. 3. Make loss and metrics configurable.
        model.compile(optimizer=get_optimizer(cfg),
                      loss=get_loss(cfg), metrics=metrics)
        checkpoint = ModelCheckpoint(cfg["train"]["model_name"] + "{epoch:02d}.h5", monitor=monitor,
                                     save_best_only=cfg["train"]["save_best_only"],
                                     save_freq='epoch')
        history = model.fit(train_gen, validation_data=valid_gen, steps_per_epoch=len(train_gen),
                            epochs=cfg["train"]["epochs"], callbacks=[checkpoint])
        show_history(history, validation)
        logging.info(f"{lgr}: Total Training Time: [{time() - start_time}] seconds")

        if test_gen is not None:
            logging.info(f"{lgr}: Starting testing.")
            results = model.evaluate(test_gen, batch_size=1, steps=test_gen.get_x_len())
            log_test_results(eval_list, results)
    else:
        logging.error(f"{lgr}: Invalid model_type. Aborting training process.")

