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


# Source: https://medium.com/@rekalantar/step-by-step-tutorial-liver-segmentation-on-ct-scans-using-tensorflow-d27bc61fbfe2
def dice_coef_single_label(y_true, y_pred, smooth=1.):
    """
  Dice = (2*|X & Y|)/ (|X|+ |Y|)
        =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
  ref: https://arxiv.org/pdf/1606.04797v1.pdf
  """
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)


def dice(y_true, y_pred, smooth=1e-7):
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)

    loss = 1 - dice_coef(y_true, y_pred, smooth=smooth)
    return loss
