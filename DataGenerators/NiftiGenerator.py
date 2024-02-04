import nibabel as nib
import numpy as np
import tensorflow as tf
from Util.Preprocessing import normalize_img, random_patch_3D
from Util.Utils import str_to_tuple


class Nifti3DGenerator(tf.keras.utils.Sequence):
    def __init__(self, cfg, x, y):
        self.x = x  # Image Set or NIFTI absolute filepaths
        self.y = y  # Label Set or Lesion absolute filepaths
        self.x_len = len(x)
        self.batch_size = cfg["train"]["batch_size"] # if patching will be used as number of patches to be extracted
        self.image_shape = str_to_tuple(cfg["data"]["image_shape"]) # if patching allowed will be used as patch size
        self.normalize = cfg["data"]["apply_norm"]
        self.shuffle = cfg["data"]["shuffle"]
        self.patching = True

    def __len__(self):
        _len = int(np.floor(self.x_len / float(self.batch_size)))
        if self.patching:
            _len = self.x_len
        return _len

    def on_epoch_end(self):
        datapoints = list(zip(self.x, self.y))
        if self.shuffle:
            np.random.shuffle(datapoints)
        self.x, self.y = map(list, zip(*datapoints))

    def __getitem__(self, idx):
        if self.patching:
            return self.extract_patches(idx)

        return self.load_batch(idx)

    def load_batch(self, idx):
        images, labels = np.zeros((self.batch_size, *self.image_shape)), \
                         np.zeros((self.batch_size, *self.image_shape))

        start_index = idx * self.batch_size
        end_index = start_index + self.batch_size

        batch_x = self.x[start_index: end_index]
        batch_y = self.y[start_index: end_index]

        for i, (img_path, lbl_path) in enumerate(zip(batch_x, batch_y)):
            img = (nib.load(img_path).get_fdata())
            lbl = (nib.load(lbl_path).get_fdata())

            assert img.shape == lbl.shape, "Loaded image's shape != loaded label's shape"

            if self.normalize:
                img = normalize_img(img)

            images[i, :, :, :] = img
            labels[i, :, :, :] = lbl

        return images.astype(np.float32), labels.astype(np.float32)

    def extract_patches(self, idx):
        images, labels = np.zeros((self.batch_size, *self.image_shape)), \
                         np.zeros((self.batch_size, *self.image_shape))

        img = nib.load(self.x[idx]).get_fdata()
        lbl = nib.load(self.y[idx]).get_fdata()

        if self.normalize:
            img = normalize_img(img)

        for i in range(0, self.batch_size):
            img_patch, lbl_patch = random_patch_3D(img, lbl, self.image_shape)
            images[i, :, :, :] = img_patch.astype(np.float32)
            labels[i, :, :, :] = lbl_patch.astype(np.float32)

        return images.astype(np.float32), labels.astype(np.float32)
