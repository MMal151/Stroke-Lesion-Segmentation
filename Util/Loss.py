import logging
import tensorflow as tf
from keras_unet_collection.losses import tversky_coef, tversky, focal_tversky

from Util.Metrics import dice_coef

CLASS_NAME = "[Util/Loss]"
CUSTOM_LOSS_FUNCTIONS = "dice,tverskycoef,tversky,ftversky"

parallel_process = False


def get_loss(cfg):
    lgr = CLASS_NAME + "[get_loss()]"
    loss = cfg["train"]["loss"].lower()

    if cfg["common_config"]["parallel_process"]:
        global parallel_process
        parallel_process = True

    if loss == 'dice':
        logging.debug(f"{lgr}: Using Dice Function as Loss Function.")
        return dice
    elif loss == 'tverskycoef':
        logging.debug(f"{lgr}: Using Tversky Coef as Loss Function.")
        return tversky_coef
    elif loss == "tversky":
        logging.debug(f"{lgr}: Using Tversky as Loss Function.")
        return tversky
    elif loss == "ftversky":
        logging.debug(f"{lgr}: Using Focal Tversky as Loss Function.")
        return focal_tversky

    logging.debug(f"{lgr}: Using {loss} as Loss Function.")
    return loss


@tf.function
def dice(y_true, y_pred, smooth=1e-7):
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)

    loss = 1 - dice_coef(y_true, y_pred, smooth)

    return loss
