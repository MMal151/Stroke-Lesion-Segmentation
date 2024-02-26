import logging
import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
from keras_unet_collection.losses import tversky_coef, tversky, focal_tversky

from Util.Metrics import dice_coef

CLASS_NAME = "[Util/Loss]"
CUSTOM_LOSS_FUNCTIONS = "dice, tverskycoef, tversky, ftversky, bce_dice"


def get_loss(loss):
    lgr = CLASS_NAME + "[get_loss()]"

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
    elif loss == 'bce_dice':
        logging.debug(f"{lgr}: Using sum of binary cross-entropy and dice as Loss Function.")
        return bce_dice_loss

    logging.debug(f"{lgr}: Using {loss} as Loss Function.")
    return loss


@tf.function
def dice(y_true, y_pred, smooth=1e-7):
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)

    return tf.reduce_mean(1 - dice_coef(y_true, y_pred, smooth))


@tf.function
def bce_dice_loss(y_true, y_pred):
    bce = BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    return tf.reduce_mean(bce(y_true, y_pred) + dice(y_true, y_pred))
