import numpy as np
import tensorflow as tf
import nibabel as nib
from Util.Utils import str_to_tuple


class Nifti3DGenerator(tf.keras.utils.Sequence):
    def __init__(self, cfg, x, y):
        self.x = x  # Image Set or NIFTI absolute filepaths
        self.y = y  # Label Set or Lesion absolute filepaths
        self.x_len = len(x)
        self.batch_size = cfg["train"]["batch_size"]
        self.image_shape = str_to_tuple(cfg["data"]["image_shape"])

    def __len__(self):
        return int(np.ceil(self.x_len / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        X = np.zeros((self.batch_size, *self.image_shape))
        Y = np.zeros((self.batch_size, *self.image_shape))

        for i, (img_file, lbl_file) in enumerate(zip(batch_x, batch_y)):
            img = nib.load(img_file).get_fdata()
            lbl = nib.load(lbl_file).get_fdata()

            # TO-DO: Need to add normalization & bias correction

            X[i, :, :, :] = img
            Y[i, :, :, :] = lbl

        return X, Y


