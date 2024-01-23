import logging

from keras_unet_collection.activations import GELU, Snake
from tensorflow.keras import models
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import L1

from Util.Utils import get_filters, str_to_tuple

CLASS_NAME = "[Model/Vnet]"

act = 'prelu'


def init_activation(activation):
    global act
    lgr = CLASS_NAME + "[init_activation()]"
    if activation != "" and len(activation) > 1:
        act = activation
    else:
        logging.error(f"{lgr} No valid activation function is defined hence using LeakyReLU as "
                      f"default activation function.")


def Activation(x):
    lgr = CLASS_NAME + "[Activation()]"
    if act == 'leakyrelu':
        return LeakyReLU()(x)
    elif act == 'relu':
        return ReLU(x)
    elif act == 'prelu':
        return PReLU()(x)
    elif act == 'gelu':
        return GELU()(x)
    elif act == 'snake':
        return Snake()(x)
    else:
        logging.error(f"{lgr} Invalid value given as activation function, using leakyReLU as default value.")
        return LeakyReLU()(x)


# Input: x -> Tensor currently under processing;
#        filters -> Number of filters to be applied in each layer;
#        num_conv_blocks -> Total number of 'convolutional blocks';
#                           convolutional blocks: Con3D Layer followed by a BatchNormalization Layer
# Returns: x -> convolutional layer;
#          x_down -> down-sampled layers
def down_block(y, filters, num_conv_blocks=1, down_sample=True):
    x = y  # Saving the state of the input tensor to ensure that it can be added later.

    for i in range(0, num_conv_blocks):
        x = Conv3D(filters, 5, padding='same', kernel_regularizer=L1(l1=0.01))(x)
        x = BatchNormalization()(x)  # Original Model doesn't support batch normalization
        x = Activation(x)  # Original Model -> PReLu

    y = Conv3D(filters, (1, 1, 1), padding='same')(y)
    x = x + y

    # TO-DO: Add option to replace this layer with max_pooling
    if down_sample:
        return x, down_sampling(x, filters)
    else:
        return x, None


def down_sampling(x, filters):
    x = Conv3D(filters, 2, 2, padding='same', kernel_regularizer=L1(l1=0.01))(x)
    x = BatchNormalization()(x)
    x = Activation(x)

    return x


# 3D-Unet Up Sampling
def up_block(x, rc, filters, num_conv_blocks=1, use_transpose=True):

    if use_transpose:
        # Does up-sampling using learnable parameters. Adaptable
        x = Conv3DTranspose(filters, kernel_size=2, strides=2, padding='same', kernel_regularizer=L1(l1=0.01))(x)
    else:
        # Does up-sampling by applying different mathematical functions. Fixed algorithm.
        x = UpSampling3D()(x)

    # For input images that are not in power of 2, the upsampling may cause a difference in shape. Adding cropping to fix this.
    if x.shape[1:-1] != rc.shape[1:-1]:
        # Calculate the amount of cropping needed for each dimension
        crop_depth = max(0, x.shape[1] - rc.shape[1])
        crop_height = max(0, x.shape[2] - rc.shape[2])
        crop_width = max(0, x.shape[3] - rc.shape[3])

        # Apply cropping
        x = Cropping3D(cropping=((0, crop_depth), (0, crop_height), (0, crop_width)))(x)

    y = x  # Saving the state of the input tensor to be added later
    x = Concatenate(axis=-1)([x, rc])

    for i in range(0, num_conv_blocks):
        x = Conv3D(filters, 3, padding='same', kernel_regularizer=L1(l1=0.01))(x)
        x = BatchNormalization()(x)
        x = Activation(x)

    y = Conv3D(filters, (1, 1, 1), padding='same')(y)
    x = Add()([y, x])

    return x


class Vnet:
    def __init__(self, cfg):
        lgr = CLASS_NAME + "[init()]"
        self.input_shape = (*str_to_tuple(cfg["data"]["image_shape"]), 1)
        self.output_classes = cfg["data"]["output_classes"]
        self.dropout = cfg["train"]["dropout"]
        self.filters = get_filters(cfg["data"]["min_filter"], 5)
        init_activation(cfg["train"]["activation"].lower())

        num_encoder_blocks = cfg["model_specific"]["vnet"]["num_encoder_blocks"].split(",")

        if len(num_encoder_blocks) >= 4:
            self.num_encoder_blocks = [int(i) for i in num_encoder_blocks]
        else:
            logging.error(f"{lgr}: Invalid number of encoder blocks configured. The total number of blocks should"
                          f"be 4. Default value of 1,2,3,3 will be used. ")
            self.num_encoder_blocks = [1, 2, 3, 3]

        num_decoder_blocks = cfg["model_specific"]["vnet"]["num_decoder_blocks"].split(",")
        if len(num_decoder_blocks) >= 3:
            self.num_decoder_blocks = [int(i) for i in num_decoder_blocks]
        else:
            logging.error(f"{lgr}: Invalid number of decoder blocks configured. The total number of blocks should"
                          f"be at least 3. Default value of 3,3,2 will be used. ")
            self.num_decoder_blocks = [3, 3, 2]

        self.use_transpose = cfg["model_specific"]["vnet"]["use_transpose"]

        self.print_info()

        # TO-DO: Need to make activation function configurable.

    def generate_model(self):
        img = Input(shape=self.input_shape)

        # Encoder
        rc_1, x = down_block(img, self.filters[0], self.num_encoder_blocks[0])
        x = Dropout(self.dropout)(x)
        rc_2, x = down_block(x, self.filters[1], self.num_encoder_blocks[1])
        x = Dropout(self.dropout)(x)
        rc_3, x = down_block(x, self.filters[2], self.num_encoder_blocks[2])
        x = Dropout(self.dropout)(x)
        rc_4, x = down_block(x, self.filters[3], self.num_encoder_blocks[3])
        x = Dropout(self.dropout)(x)

        # Bottleneck layer
        x, _ = down_block(x, self.filters[4], self.num_encoder_blocks[3], False)

        # Decoder
        x = up_block(x, rc_4, self.filters[4], self.num_decoder_blocks[0], self.use_transpose)
        x = Dropout(self.dropout)(x)
        x = up_block(x, rc_3, self.filters[3], self.num_decoder_blocks[1], self.use_transpose)
        x = Dropout(self.dropout)(x)
        x = up_block(x, rc_2, self.filters[2], self.num_decoder_blocks[2], self.use_transpose)
        x = Dropout(self.dropout)(x)
        x = up_block(x, rc_1, self.filters[1], 1, self.use_transpose)

        x = Dropout(self.dropout)(x)  # Not included in the original model.

        # Update the output layer based on the number of classes
        output = Conv3D(self.output_classes, 1, activation='sigmoid')(x)

        out = models.Model(img, output, name='vnet')
        out.summary()
        return out

    def print_info(self):
        logging.info("[Model/Vnet][print_info()] Model initialized using the following configurations.")
        logging.info(self.__dict__)
