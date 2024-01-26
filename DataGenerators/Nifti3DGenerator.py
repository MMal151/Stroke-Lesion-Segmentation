import logging

import numpy as np
import tensorflow as tf
import nibabel as nib
from scipy.ndimage import zoom
from tensorflow.data.experimental import AUTOTUNE

from Util.Preprocessing import normalize_img
from Util.Utils import str_to_tuple

CLASS_NAME = "[DataGenerators/Nifti3DGenerator]"


class Nifti3DGenerator(tf.keras.utils.Sequence):
    def __init__(self, cfg, x, y):
        self.x = x  # Image Set or NIFTI absolute filepaths
        self.y = y  # Label Set or Lesion absolute filepaths
        self.x_len = len(x)
        self.batch_size = cfg["train"]["batch_size"]
        self.image_shape = str_to_tuple(cfg["data"]["image_shape"])
        self.aply_norm = cfg["data"]["apply_norm"]
        self.shuffle = cfg["data"]["shuffle"]
        self.iterator = None
        self.create_dataset()

    def __len__(self):
        return int(np.floor(self.x_len / float(self.batch_size)))

    def on_epoch_end(self):
        datapoints = list(zip(self.x, self.y))
        if self.shuffle:
            np.random.shuffle(datapoints)
        self.x, self.y = map(list, zip(*datapoints))
        # Dataset is created again to ensure the state of dataset at the end of the epoch matches the new state.
        self.create_dataset()

    def create_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.x, self.y))
        dataset = (dataset.shuffle(self.x_len)
                   .map(lambda x, y: tf.numpy_function(self.load_data, [x, y], [tf.float32, tf.float32]),
                        num_parallel_calls=AUTOTUNE)
                   .repeat()
                   .batch(self.batch_size)
                   .prefetch(AUTOTUNE))
        self.iterator = iter(dataset)

    def load_data(self, img_path, lbl_path):
        lgr = CLASS_NAME + "[load_data()]"
        img = nib.load(img_path.decode()).get_fdata()  # Tensorflow dataset encodes all values.
        lbl = nib.load(lbl_path.decode()).get_fdata()

        assert img.shape == lbl.shape, "Loaded image's shape != loaded label's shape"

        if img.shape != self.image_shape:
            logging.debug(f"{lgr}: Down-sampling images & labels to match the configured dimensions. "
                              f"Original Image Dimensions: {img.shape} \n"
                              f"Original Label Dimensions: {lbl.shape}")
            img = zoom(img, self.image_shape / np.array(img.shape)).astype(np.float32)
            lbl = zoom(lbl, self.image_shape / np.array(lbl.shape), order=0).astype(np.float32)

        if self.aply_norm:
            img = normalize_img(img)

        return img.astype(np.float32), lbl.astype(np.float32)

    def __getitem__(self, idx):
        return self.iterator.get_next()
