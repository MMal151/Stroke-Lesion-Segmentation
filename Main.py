import logging
import os
import tensorflow as tf
import time

from Process.Inference import get_segmentation
from Process.Test_Temp import test
from Process.Train import train
from Process.Utilities import print_train_configurations, print_test_configurations
from Util.Utils import get_configurations

CLASS_NAME = "[Main]"


def configure_logger(filename, level=""):
    log_level = logging.INFO

    if level == "debug":
        log_level = logging.DEBUG
    elif level == "warn":
        log_level = logging.WARN

    logging.basicConfig(filename=filename + str(time.strftime('%Y%m%dT%H%M')) + ".log", encoding='utf-8',
                        level=log_level,
                        format='%(asctime)s %(message)s')


def configure_gpus(cfg):
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg["misc"]["gpu"]["no_gpus"]

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
    if cfg["misc"]["gpu"]["alw_para_prs"]:
        opt = cfg["misc"]["gpu"]["strategy"]

        if opt == "mirrored":
            strategy = tf.distribute.MirroredStrategy()

    return strategy


if __name__ == "__main__":

    config = get_configurations("config.yml")

    configure_logger(config["misc"]["logging"]["filename"], config["misc"]["logging"]["level"])

    if config["misc"]["mode"] == "train":
        print_train_configurations(config)
        train(config, configure_gpus(config))
    elif config["misc"]["mode"] == "test":
        print_test_configurations(config)
        test(config)
    elif config["misc"]["mode"] == "inference":
        get_segmentation(config)
