import logging
from time import time

from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from DatasetLoader import BalancedLoader, DefaultLoader
from Util.Loss import get_loss
from Util.Metrics import get_metrics
from Util.Optimizers import get_optimizer
from Util.Preprocessing import data_augmentation
from Util.Utils import get_all_possible_subdirs, remove_dirs
from Util.Visualization import show_history
from DataGenerators.NiftiGenerator import Nifti3DGenerator
from Model.Unet3D import Unet3D
from Model.Vnet import Vnet
from Process.Utils import load_data, load_model

CLASS_NAME = "[Process/Train]"

def train(cfg, strategy=None):
    lgr = CLASS_NAME + "[train()]"
    logging.info(f"{lgr}: Starting Training.")

    if cfg["train"]["data"]["loader"] == "balanced":
        x_train, y_train, x_valid, y_valid, x_test, y_test = BalancedLoader.load_dataset(cfg)
    else:
        x_train, y_train, x_valid, y_valid, x_test, y_test = DefaultLoader.loader(cfg)

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

        callbacks = []

        checkpoint = ModelCheckpoint(cfg["train"]["save"]["model_name"] + "{epoch:02d}.h5", monitor=monitor,
                                     save_best_only=cfg["train"]["save"]["best_only"],
                                     save_freq='epoch')
        callbacks.append(checkpoint)

        if cfg["train"]["aply_early_stpng"]:
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, min_delta=0.005, mode="min",
                                           restore_best_weights=True)
            callbacks.append(early_stopping)

        history = model.fit(train_gen, validation_data=valid_gen, steps_per_epoch=len(train_gen),
                            epochs=cfg["train"]["epochs"], callbacks=callbacks)
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

    train_gen = Nifti3DGenerator(cfg, x_train, y_train, True, "train")
    valid_gen, test_gen = None, None
    if len(x_valid) > 0:
        valid_gen = Nifti3DGenerator(cfg, x_valid, y_valid, False, "valid")
    if len(x_test) > 0:
        test_gen = Nifti3DGenerator(cfg, x_test, y_test, False, "test")

    logging.info(f"{lgr}: Generating Model.")

    return train_gen, valid_gen, test_gen


def log_test_results(eval_list, results):
    lgr = CLASS_NAME + "[log_test_results()]"
    logging.info(f"{lgr}: Loss = [{results[0]}]")

    for i, metric in enumerate(eval_list, start=1):
        logging.info(f"{lgr}: {metric} = [{results[i]}]")
