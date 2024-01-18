import logging
import tensorflow as tf
from keras_unet_collection.activations import GELU, Snake

from Util.Utils import get_all_possible_files_paths, is_valid_file
from Util.Loss import CUSTOM_LOSS_FUNCTIONS, get_loss
from Util.Metrics import get_metrics_test

CLASS_NAME = "[Process/Utilities]"
CUSTOM_ACTIVATIONS = "gelu,snake"


def load_data(input_path, img_ext, lbl_ext):
    lgr = CLASS_NAME + "[load_data()]"

    logging.info(f"{lgr}: Loading Dataset.")

    # Loading dataset
    images = get_all_possible_files_paths(input_path, img_ext)
    labels = get_all_possible_files_paths(input_path, lbl_ext)

    logging.debug(f"{lgr}: Loaded Images: {images} \n Loaded Labels: {labels}")
    return images, labels


# Custom Objects includes all custom activations, loss functions, and performance metrics.
def get_custom_objects(cfg):
    lgr = CLASS_NAME + "[get_custom_objects()]"
    custom_objects = get_metrics_test(cfg["train"]["perf_metrics"])
    if CUSTOM_ACTIVATIONS.__contains__(cfg["train"]["activation"].lower()):
        logging.debug(f"{lgr}: Adding custom activation function in custom_obj list.")
        if cfg["train"]["activation"].lower() == "gelu":
            custom_objects["GELU"] = GELU()
        elif cfg["train"]["activation"].lower() == "snake":
            custom_objects["Snake"] = Snake()

    if CUSTOM_LOSS_FUNCTIONS.__contains__(cfg["train"]["loss"].lower()):
        logging.debug(f"{lgr}: Adding custom loss function in custom_obj list.")
        custom_objects[cfg["train"]["loss"].lower()] = get_loss(cfg)

    return custom_objects


def load_model(cfg, resume_train):
    model = None
    if resume_train:
        model_file = cfg["train"]["model_file"]
    else:
        model_file = cfg["test"]["model_load_state"]

    if is_valid_file(model_file):
        model = tf.keras.models.load_model(model_file,
                                           custom_objects=get_custom_objects(cfg))

    return model
