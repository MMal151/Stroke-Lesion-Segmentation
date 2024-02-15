import logging

import nibabel as nib
import numpy as np
import tensorflow as tf

from Process.Utils import generate_patch_idx
from Util.Preprocessing import normalize_img, random_patch_3D
from Util.Utils import str_to_tuple

CLASS_NAME = "[DataGenerators/NiftiGenerator]"


class Nifti3DGenerator(tf.keras.utils.Sequence):
    def __init__(self, cfg, x, y):
        self.x = x  # Image Set or NIFTI absolute filepaths
        self.y = y  # Label Set or Lesion absolute filepaths

        self.patching = False

        if cfg["train"]["data"]["patch"]["alw_patching"]:
            self.patching = True
            self.patch_shape = str_to_tuple(cfg["train"]["data"]["patch"]["patch_size"])
            self.total_patches = cfg["train"]["data"]["patch"]["total_patches"]
            self.random_patch = cfg["train"]["data"]["patch"]["random_patches"]

            if not cfg["train"]["data"]["patch"]["random_patches"]:
                self.patch_idx = []

        self.image_shape = str_to_tuple(cfg["train"]["data"]["image_shape"])
        self.batch_size = cfg["train"]["data"]["batch_size"]
        self.normalize = cfg["train"]["data"]["norm_data"]
        self.shuffle = cfg["train"]["data"]["shuffle"]

        self.step_per_epoch = cfg["train"]["num_iter"]
        self.init_dataset()

    def __len__(self):
        if self.step_per_epoch > 0:
            return self.step_per_epoch
        return int(np.floor(len(self.x) / float(self.batch_size)))

    def on_epoch_end(self):
        datapoints = list(zip(zip(self.x, self.y), self.patch_idx))
        if self.shuffle:
            np.random.shuffle(datapoints)
        xy, self.patch_idx = map(list, zip(*datapoints))
        self.x, self.y = zip(*xy)

    def __getitem__(self, idx):
        return self.load_batch(idx)

    def load_batch(self, idx):

        image_shape = self.image_shape

        if self.patching:
            image_shape = self.patch_shape

        images, labels = np.zeros((self.batch_size, *image_shape)), \
                         np.zeros((self.batch_size, *image_shape))

        start_index = idx * self.batch_size
        end_index = start_index + self.batch_size

        batch_x = self.x[start_index: end_index]
        batch_y = self.y[start_index: end_index]

        if self.patching and not self.random_patch:
            batch_idx = self.patch_idx[start_index: end_index]

        for i, (img_path, lbl_path) in enumerate(zip(batch_x, batch_y)):
            img = (nib.load(img_path).get_fdata())
            lbl = (nib.load(lbl_path).get_fdata())

            assert img.shape == lbl.shape, "Loaded image's shape != loaded label's shape"

            if self.normalize:
                img = normalize_img(img)

            if self.patching:
                if self.random_patch:
                    img, lbl = random_patch_3D(img, lbl, image_shape)
                else:
                    (ax_1, ax_2, ax_3) = batch_idx[i]
                    img = img[ax_1: ax_1 + image_shape[0], ax_2: ax_2 + image_shape[1], ax_3: ax_3 + image_shape[2]]
                    lbl = lbl[ax_1: ax_1 + image_shape[0], ax_2: ax_2 + image_shape[1], ax_3: ax_3 + image_shape[2]]

            images[i, :, :, :] = img.astype(np.float32)
            labels[i, :, :, :] = lbl.astype(np.float32)

        return images.astype(np.float32), labels.astype(np.float32)

    def init_dataset(self):
        lgr = CLASS_NAME + "[init_data()]"
        if self.patching:
            if self.random_patch:
                logging.debug(f"{lgr}: Since random patching in enabled, each training data point will be used to extract"
                              f" {self.total_patches} patches.")
                # Extending each data point as many times as the number of patches,
                # this method was adopted to limit the memory and also to ensure that the patches of the same data-point
                # will not be a part of the same batch.
                self.x.extend(self.x * self.total_patches)
                self.y.extend(self.y * self.total_patches)
            else:
                logging.debug(f"{lgr}: Patching is enabled, patches w.r.t stride [{self.total_patches}] will be"
                              f"extracted from each image.")
                no_patches, self.patch_idx = generate_patch_idx(self.image_shape, self.total_patches,
                                                                self.patch_shape, len(self.x))
                self.x.extend(self.x * no_patches)
                self.y.extend(self.y * no_patches)
        self.on_epoch_end()
