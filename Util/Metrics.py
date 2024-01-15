import logging

from keras_unet_collection.losses import dice_coef
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


