import logging
from tensorflow.keras import models
from tensorflow.keras.layers import *

from Util.Utils import get_filters, str_to_tuple


# 3D-Unet Down Sampling
def down_block(x, filters, use_maxpool=True):
    x = Conv3D(filters, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv3D(filters, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    if use_maxpool:
        return MaxPooling3D(strides=(2, 2, 2), padding='same')(x), x
    else:
        return x


# 3D-Unet Up Sampling
def up_block(x, y, filters, use_transpose=True):
    if use_transpose:
        # Does up-sampling using learnable parameters. Adaptable
        x = Conv3DTranspose(filters, kernel_size=2, strides=2, padding='same')(x)
    else:
        # Does up-sampling by applying different mathematical functions. Fixed algorithm.
        x = UpSampling3D()(x)

    # For input images that are not in power of 2, the upsampling may cause a difference in shape. Adding cropping to fix this.
    if x.shape[1:-1] != y.shape[1:-1]:
        # Calculate the amount of cropping needed for each dimension
        crop_depth = max(0, x.shape[1] - y.shape[1])
        crop_height = max(0, x.shape[2] - y.shape[2])
        crop_width = max(0, x.shape[3] - y.shape[3])

        # Apply cropping
        x = Cropping3D(cropping=((0, crop_depth), (0, crop_height), (0, crop_width)))(x)

    x = Concatenate(axis=-1)([x, y])
    x = Conv3D(filters, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv3D(filters, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    return x


class Unet3D:
    def __init__(self, cfg):
        self.input_shape = (*str_to_tuple(cfg["data"]["image_shape"]), 1)
        self.output_classes = cfg["data"]["output_classes"]
        self.dropout = cfg["train"]["dropout"]
        self.filters = get_filters(cfg["data"]["min_filter"], 5)
        self.print_info()

        # TO-DO: Need to make activation function configurable.

    def generate_model(self):
        img = Input(shape=self.input_shape)
        x, temp1 = down_block(img, self.filters[0])
        x, temp2 = down_block(x, self.filters[1])
        x, temp3 = down_block(x, self.filters[2])
        x, temp4 = down_block(x, self.filters[3])
        x = down_block(x, self.filters[4], use_maxpool=False)

        # decode
        x = up_block(x, temp4, self.filters[4])
        x = up_block(x, temp3, self.filters[2])
        x = up_block(x, temp2, self.filters[1])
        x = up_block(x, temp1, self.filters[0])

        x = Dropout(self.dropout)(x)

        # Update the output layer based on the number of classes
        if self.output_classes == 1:
            output = Conv3D(1, 1, activation='sigmoid')(x)
        else:
            output = Conv3D(self.output_classes, 1, activation='softmax')(x)

        out = models.Model(img, output, name='unet_3d')
        out.summary()
        return out

    def print_info(self):
        logging.info("[Model/Unet3D][print_info()] Model initialized using the following configurations.")
        logging.info(self.__dict__)

