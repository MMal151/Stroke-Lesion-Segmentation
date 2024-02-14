import logging
from time import time

from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint

from DataGenerators.NiftiGenerator import Nifti3DGenerator
from Model.Unet3D import Unet3D
from Model.Vnet import Vnet
from Process.Util import load_data, load_model
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

    logging.debug(f"{lgr}: x_train: {x_train} \n y_train: {y_train} \n x_{div_type}: {x_valid} \n "
                  f"y_{div_type}: {y_valid}")
    logging.info(f"{lgr}: Training-Instances: {len(x_train)}. "
                 f"{div_type}-Instances: {len(x_valid)}")

    return x_train, x_valid, y_train, y_valid


def train(cfg, strategy=None):
    lgr = CLASS_NAME + "[train()]"
    logging.info(f"{lgr}: Starting Training.")

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

    params = {"x_train": x_train, "y_train": y_train, "x_valid": x_valid, "y_valid": y_valid, "x_test": x_test,
              "y_test": y_test}

    if strategy is not None:
        with strategy.scope():
            train_gen, valid_gen, test_gen = get_generators(cfg, **params)
            fit_model(cfg, train_gen, valid_gen, test_gen)
    else:
        train_gen, valid_gen, test_gen = get_generators(cfg, **params)
        fit_model(cfg, train_gen, valid_gen, test_gen)


def fit_model(cfg, train_gen, valid_gen, test_gen):
    lgr = CLASS_NAME + "[fit_model()]"

    start_time = time()
    model = None
    if cfg["train"]["resume"]["resume_train"]:
        model = load_model(cfg, True)

    if cfg["train"]["model_type"].lower() == "unet" and model is None:
        model = Unet3D(cfg).generate_model()
    elif cfg["train"]["model_type"].lower() == "vnet" and model is None:
        model = Vnet(cfg).generate_model()

    if model is not None:
        monitor = 'loss'
        validation = False
        if valid_gen is not None:
            monitor = "val_loss"
            validation = True

        metrics, eval_list = get_metrics(cfg["train"]["perf_metrics"])
        model.compile(optimizer=get_optimizer(cfg),
                      loss=get_loss(cfg["train"]["loss"].lower()), metrics=metrics)
        checkpoint = ModelCheckpoint(cfg["train"]["save"]["model_name"] + "{epoch:02d}.h5", monitor=monitor,
                                     save_best_only=cfg["train"]["save"]["best_only"],
                                     save_freq='epoch')

        history = model.fit(train_gen, validation_data=valid_gen, steps_per_epoch=len(train_gen),
                            epochs=cfg["train"]["epochs"], callbacks=[checkpoint])
        show_history(history, validation)
        logging.info(f"{lgr}: Total Training Time: [{time() - start_time}] seconds")

        if test_gen is not None:
            logging.info(f"{lgr}: Starting testing.")
            results = model.evaluate(test_gen, batch_size=1, steps=len(test_gen))
            log_test_results(eval_list, results)
    else:
        logging.error(f"{lgr}: Invalid model_type. Aborting training process.")


def get_generators(cfg, x_train, y_train, x_valid, y_valid, x_test, y_test):
    lgr = CLASS_NAME + "[get_generators()]"
    logging.info(f"{lgr}: Creating Generators for training (& validation & test) data.")

    train_gen = Nifti3DGenerator(cfg, x_train, y_train)
    valid_gen, test_gen = None, None
    if len(x_valid) > 0:
        valid_gen = Nifti3DGenerator(cfg, x_valid, y_valid)
    if len(x_test) > 0:
        test_gen = Nifti3DGenerator(cfg, x_test, y_test)

    logging.info(f"{lgr}: Generating Model.")

    return train_gen, valid_gen, test_gen


def log_test_results(eval_list, results):
    lgr = CLASS_NAME + "[log_test_results()]"
    logging.info(f"{lgr}: Loss = [{results[0]}]")

    for i, metric in enumerate(eval_list, start=1):
        logging.info(f"{lgr}: {metric} = [{results[i]}]")
