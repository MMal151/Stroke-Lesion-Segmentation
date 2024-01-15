import logging
import os
import random
import nibabel as nib
import numpy as np
import tensorflow as tf
from keras_unet_collection.activations import GELU, Snake

from DataGenerators.Nifti3DGenerator import Nifti3DGenerator
from Process.Utilities import load_data
from Util.Loss import dice_coef_single_label, CUSTOM_LOSS_FUNCTIONS, get_loss
from Util.Metrics import get_metrics_test, get_metrics
from Util.Utils import is_valid_file, is_valid_dir, str_to_tuple

CLASS_NAME = "[Process/Test]"
CUSTOM_ACTIVATIONS = "gelu,snake"


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


# Before testing, the data should be cropped and skull stripped to match the size of the input.
def test(cfg):
    lgr = CLASS_NAME + "[test()]"
    if is_valid_file(cfg["test"]["model_load_state"]):
        model = tf.keras.models.load_model(cfg["test"]["model_load_state"],
                                           custom_objects=get_custom_objects(cfg))

        x_test, y_test = load_data(cfg["data"]["input_path"], cfg["data"]["img_ext"],
                                   cfg["data"]["lbl_ext"])

        test_gen = Nifti3DGenerator(cfg, x_test, y_test)

        results = model.evaluate(test_gen, batch_size=1, steps=test_gen.get_x_len())
        _, eval_list = get_metrics(cfg["train"]["perf_metrics"])
        log_test_results(eval_list, results)
        if cfg["test"]["save_random_samples"]:
            logging.info(f"{lgr}: Saving test results for some samples in /Test_Results.")
            random_idx = set()  # Used to store indexes already processed, to ensure same datapoints are not processed again.

            num_samples = 5
            if cfg["test"]["num_samples"] > 0:
                num_samples = cfg["test"]["num_samples"]
            else:
                logging.info(f"{lgr}: Invalid number of samples configured. Using default value of 5.")

            for i in range(0, num_samples):
                idx = random.randint(0, len(x_test))

                if not random_idx.__contains__(idx):
                    image_shape = str_to_tuple(cfg["data"]["image_shape"])
                    # Loading and re-shaping image to match the model's input layer's shape.
                    curr_img = (nib.load(x_test[idx]).get_fdata()).reshape(1, image_shape[0], image_shape[1],
                                                                           image_shape[2], 1)
                    curr_label = (nib.load(x_test[idx]).get_fdata()).reshape(1, image_shape[0], image_shape[1],
                                                                             image_shape[2], 1).astype(np.float32)
                    predict = model.predict(curr_img)
                    dice = dice_coef_single_label(curr_label, predict)
                    logging.info(f"{lgr}: Predicted Dice Coeff for {x_test[idx]}: {dice}")

                    if not is_valid_dir("Test_Results"):
                        os.mkdir("Test_Results/")

                    # y_test contains the absolute path of mask. Firstly dividing the path to get just the filename
                    # and then dividing the filename to add "_Pred.nii.gz"
                    predicted_segmentation = predict.reshape(image_shape[0], image_shape[1], image_shape[2], 1)

                    save_name = y_test[idx].split('/')[-1].split('.nii.gz')[0] + "_Pred.nii.gz"
                    save_path = os.path.join("Test_Results", save_name)
                    ni = nib.Nifti1Image(predicted_segmentation, affine=np.eye(4))
                    nib.save(ni, save_path)
                    random_idx.add(idx)
                else:
                    i = i - 1  # Adding this to ensure number of samples remains the same.
    else:
        logging.info(f"{lgr}: Invalid Model Filename/Path. Need valid name to perform testing.")


def log_test_results(eval_list, results):
    lgr = CLASS_NAME + "[log_test_results()]"
    logging.info(f"{lgr}: Loss = [{results[0]}]")

    for i, metric in enumerate(eval_list, start=1):
        logging.info(f"{lgr}: {metric} = [{results[i]}]")
