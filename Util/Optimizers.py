import logging

import yaml
import tensorflow as tf
from tensorflow.keras.optimizers import *

from Util.Utils import get_configurations

CLASS_NAME = "[Util/Optimizers]"


def get_schedular(lr):
    # Reading Learning Rate Schedular configurations from config file.
    cfg = get_configurations("config_lrs.yml")

    lgr = CLASS_NAME + "[get_schedular()]"
    opt = cfg["learning_schedular"]["type"].lower()

    if opt == "cosine":
        decay_steps = 1000

        if cfg["learning_schedular"]["decay_steps"] > 0:
            decay_steps = cfg["learning_schedular"]["decay_steps"]

        logging.debug(f"{lgr}: Using Cosine Decay learning schedular with configurations initial_learning_rate: [{lr}]"
                      f" decay_steps: [{decay_steps}]")

        return tf.keras.optimizers.schedules.CosineDecay(lr, decay_steps)

    elif opt == "exponential":
        decay_steps = 100000
        decay_rate = 1000

        if cfg["learning_schedular"]["decay_steps"] > 0:
            decay_steps = cfg["learning_schedular"]["decay_steps"]
        if cfg["learning_schedular"]["exponential"]["decay_rate"] > -1:
            decay_rate = cfg["learning_schedular"]["exponential"]["decay_rate"]

        logging.debug(f"{lgr}: Using Exponential learning schedular with configurations initial_learning_rate: [{lr}]"
                      f" decay_steps: [{decay_steps}]"
                      f" and decay_rate: [{decay_rate}].")

        return tf.keras.optimizers.schedules.ExponentialDecay(lr, decay_steps, decay_rate)

    elif opt == "polynomial":
        decay_steps = 10000
        end_learning_rate = 0.01

        if cfg["learning_schedular"]["decay_steps"] > 0:
            decay_steps = cfg["learning_schedular"]["decay_steps"]
        if cfg["learning_schedular"]["polynomial"]["end_learning_rate"] > -1:
            end_learning_rate = cfg["learning_schedular"]["polynomial"]["end_learning_rate"]

        logging.debug(f"{lgr}: Using Polynomial learning schedular with configurations initial_learning_rate: [{lr}]"
                      f" decay_steps: [{decay_steps}]"
                      f" and end_learning_rate: [{end_learning_rate}].")

        return tf.keras.optimizers.schedules.PolynomialDecay(lr, decay_steps, end_learning_rate)

    return lr


def get_optimizer(cfg):
    lgr = CLASS_NAME + "[get_optimizer()]"

    opt, lr = cfg["train"]["optimizer"].lower(), cfg["train"]["learning_rate"]

    if cfg["train"]["aply_lr_sch"]:
        lr = get_schedular(lr)

    if opt == 'adam':
        return Adam(learning_rate=lr)
    elif opt == 'sgd':
        return SGD(learning_rate=lr)
    elif opt == 'rmsprop':
        return RMSprop(learning_rate=lr)
    elif opt == 'adadelta':
        return Adadelta(learning_rate=lr)
    elif opt == 'adamax':
        return Adamax(learning_rate=lr)
    elif opt == 'adagrad':
        return Adagrad(learnitng_rate=lr)
    else:
        logging.error(f"{lgr}: Invalid Optimizer, using Adam as default optimizer.")
        return Adam(learning_rate=lr)
