import logging

import tensorflow as tf
from tensorflow.keras.metrics import *

from Util.Utils import is_valid_str

CLASS_NAME = "[Util/Metrics]"


def get_metrics(met_list="dice_coef"):
    lgr = CLASS_NAME + "[get_metrics()]"
    metrics = []
    eval_list = []

    if is_valid_str(met_list):
        if met_list.__contains__("acc"):
            metrics.append(Accuracy(name="acc"))
            eval_list.append("Accuracy")
        if met_list.__contains__("mean_iou"):
            metrics.append(MeanIoU(num_classes=2, name="mean_iou"))
            eval_list.append("Mean IoU")
        if met_list.__contains__("recall"):
            metrics.append(Recall(name="recall"))
            eval_list.append("Recall")
        if met_list.__contains__("prec"):
            metrics.append(Precision(name="prec"))
            eval_list.append("Precision")
        if met_list.__contains__("dice_coef"):
            metrics.append(dice_coef)
            eval_list.append("Dice-Coefficient")
    else:
        logging.info(f"{lgr}: Invalid input string: [{met_list}]. Using Dice Coefficient as the default metric.")
        metrics.append(dice_coef)
        eval_list.append("Dice-Coefficient")

    return metrics, eval_list


def get_metrics_test(met_list="dice_coef"):
    lgr = CLASS_NAME + "[get_metrics_test()]"
    metrics = {}

    if is_valid_str(met_list):
        if met_list.__contains__("acc"):
            metrics["acc"] = Accuracy(name="acc")
        if met_list.__contains__("mean_iou"):
            metrics["mean_iou"] = MeanIoU(num_classes=2, name="mean_iou")
        if met_list.__contains__("recall"):
            metrics["recall"] = Recall(name="recall")
        if met_list.__contains__("prec"):
            metrics["prec"] = Precision(name="prec")
        if met_list.__contains__("dice_coef"):
            metrics["dice_coef"] = dice_coef
    else:
        logging.info(f"{lgr}: Invalid input string: [{met_list}]. Using Dice Coefficient as the default metric.")
        metrics["dice_coef"] = dice_coef

    return metrics


@tf.function
def dice_coef(y_true, y_pred, smooth=1e-7, batch=True):
    # First axis is added for ensure batches are considered for dice calculation.
    if batch:
        axis = list(range(0, len(y_true.shape)))  # Assuming tensor is channel last. Expected Tensor Shape (B, H, W, D, C)
    else:
        axis = list(range(1, len(y_true.shape)))

    # y_true_pos = tf.keras.backend.flatten(y_true)
    # y_pred_pos = tf.keras.backend.flatten(y_pred)

    true_pos = tf.reduce_sum((y_true * y_pred), axis=axis, keepdims=False)  # Sum across spatial dimensions
    false_neg = tf.reduce_sum((y_true * (1 - y_pred)), axis=axis, keepdims=False)
    false_pos = tf.reduce_sum(((1 - y_true) * y_pred), axis=axis, keepdims=False)

    n = 2.0 * true_pos + smooth
    d = 2.0 * true_pos + false_pos + false_neg + smooth
    # Need to check clip_by_value function
    return tf.reduce_mean(n / tf.clip_by_value(d, d, 1e-8))
