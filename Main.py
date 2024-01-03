import yaml
import logging
import os
import tensorflow as tf
import time

from Process.Test import test
from Process.Train import train


def configure_logger(filename, level=""):
    log_level = logging.INFO

    if level == "debug":
        log_level = logging.DEBUG
    elif level == "warn":
        log_level = logging.WARN

    logging.basicConfig(filename=filename + str(time.strftime('%Y%m%dT%H%M')) + ".log", encoding='utf-8', level=log_level,
                        format='%(asctime)s %(message)s')


def print_configurations(cfg):
    logging.debug("Following configurations were loaded.")
    logging.debug("Mode: " + cfg["common_config"]["process"])
    logging.debug("GPUs: " + cfg["common_config"]["gpus"])
    logging.debug("-------Data Configurations-------")
    logging.debug("Input Path: " + cfg["data"]["input_path"])
    logging.debug("Image Extension: " + cfg["data"]["img_ext"])
    logging.debug("Label Extension: " + cfg["data"]["lbl_ext"])
    logging.debug("Minimum Filter: " + str(cfg["data"]["min_filter"]))
    logging.debug("Image Shape: " + cfg["data"]["image_shape"])
    logging.debug("Output Classes: " + str(cfg["data"]["output_classes"]))
    logging.debug("Apply Augmentation: " + str(cfg["data"]["apply_augmentation"]))

    if cfg["data"]["apply_augmentation"]:
        logging.debug("-------Augmentation Configurations-------")
        logging.debug("Factor: " + str(cfg["augmentation"]["factor"]))
        logging.debug("Technique: " + cfg["augmentation"]["technique"])

    logging.debug("-------Model Configurations------")
    logging.debug("Validation Ratio: " + str(cfg["train"]["valid_ratio"]))
    logging.debug("Batch Size: " + str(cfg["train"]["batch_size"]))
    logging.debug("Dropout: " + str(cfg["train"]["dropout"]))
    logging.debug("Learning Rate: " + str(cfg["train"]["learning_rate"]))
    logging.debug("Epochs: " + str(cfg["train"]["epochs"]))
    logging.debug("save_best_only: " + str(cfg["train"]["save_best_only"]))

    if cfg["train"]["test_on_same_data"]:
        logging.debug("-------Testing Configurations-------")
        logging.debug("Validation Ratio: " + str(cfg["train"]["test_ratio"]))



def configure_gpus(cfg):
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg["common_config"]["gpus"]

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    tf.keras.backend.clear_session()

    return enable_parallel_process(cfg)


def enable_parallel_process(cfg):
    strategy = None
    if cfg["common_config"]["parallel_process"]:
        strategy = tf.distribute.MirroredStrategy()

    return strategy


if __name__ == "__main__":
    with open("config.yml", "r") as configFile:
        cfg = yaml.safe_load(configFile)

    configure_logger(cfg["logging_config"]["filename"], cfg["logging_config"]["level"])
    print_configurations(cfg)

    if cfg["common_config"]["process"] == "train":
        train(cfg, configure_gpus(cfg))
    elif cfg["common_config"]["process"] == "test":
        test(cfg)


