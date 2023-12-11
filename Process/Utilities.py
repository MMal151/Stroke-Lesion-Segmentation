import logging

from Util.Utils import get_all_possible_files_paths

CLASS_NAME = "[Process/Utilities]"


def load_data(input_path, img_ext, lbl_ext):
    lgr = CLASS_NAME + "[load_data()]"

    logging.info(f"{lgr}: Loading Dataset.")

    # Loading dataset
    images = get_all_possible_files_paths(input_path, img_ext)
    labels = get_all_possible_files_paths(input_path, lbl_ext)

    logging.debug(f"{lgr}: Loaded Images: {images} \n Loaded Labels: {labels}")
    return images, labels
