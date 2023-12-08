import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from sklearn.model_selection import train_test_split, KFold
from Loss import dice_coef
from Util import Utils
from DataGenerators.NiiDataGenerators import NiiDataGenerator
from Test import evaluate_test_data, evaluate_patched_test_data
from Unet3D import unet_3d
from Util.Visualization import show_history, show_history_kfolds
from argparse import ArgumentParser


def get_learning_rate_scheduler(lr, decay_steps, decay_rate):
    return ExponentialDecay(lr, decay_steps, decay_rate, staircase=True)


def get_early_stopping():
    early_stopping = EarlyStopping(
        monitor='val_dice_coef',
        patience=15,
        min_delta=0.001,
        verbose=1,
        restore_best_weights=True
    )
    return early_stopping


def get_arguments():
    parser = ArgumentParser()
    parser.add_argument('--data_path', dest='data_path', type=str, required=True)
    parser.add_argument('--train_ratio', dest='train_ratio', type=float, default=1.0)
    parser.add_argument('--test_ratio', dest='test_ratio', type=float, default=0.0)
    parser.add_argument('--valid_ratio', dest='valid_ratio', type=float, default=0.1,
                        help="Define valid_ratio as 0.0, if validation should be skipped.")
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=4)
    parser.add_argument('--classes', dest='classes', type=int, default=1)
    parser.add_argument('--lr', dest='lr', type=float, default=1e-3)
    parser.add_argument('--optimizer', dest='optimizer', type=str, default='adam')
    parser.add_argument('--dropout', dest='dropout', type=float, default=0.2)
    parser.add_argument('--seed', dest='seed', default=2023, type=int)
    parser.add_argument('--image_size', dest='image_size', default=256, type=int)
    parser.add_argument('--model_save', dest='model_save', default='Unet.h5', type=str)
    parser.add_argument('--shuffle', dest='shuffle', default=True, type=bool)
    parser.add_argument('--epochs', dest='epochs', type=int, default=20)
    parser.add_argument('--use_transpose', dest='use_transpose', default=True, type=bool,
                        help='Whether to use transpose convolutional layer or simple up-sampling layer.')
    parser.add_argument('--use_maxpool', dest='use_maxpool', default=True, type=bool,
                        help='Use max_pooling layer during down-sampling.')
    parser.add_argument('--use_kfold', dest='use_kfold', default=False, required=False, type=bool,
                        help="Use kfold for training model.")
    parser.add_argument('--folds', dest='folds', default=5, required=False, type=int,
                        help="Number of folds for k-fold. Default = 5. "
                             "Will only be checked if use_kfold is True.")
    parser.add_argument('--aply_early_stp', dest='aply_early_stp', default=False, required=False, type=bool,
                        help="Whether to apply early stepping while training. Default: False")
    parser.add_argument('--aply_normalization', dest="aply_norm", type=bool, default=False,
                        help="Apply Z-Score Normalization on the input images.")
    parser.add_argument('--test_path', dest="test_path", type=str, default=None,
                        help="Path of test set; if its different from train_set. test_ratio should be equal to zero.")
    parser.add_argument('--aply_patching', dest="aply_patch", type=bool, default=False,
                        help="Instead of resampling the input image to defined image_size, extract patches of the size defined in patch_size."
                             "image_size will be used to pad the original MRI volume to ensure all three axis are of the same size.")
    parser.add_argument('--patch_size', dest="patch_size", type=int, default=32,
                        help="Size of each patch, only used if aply_patching = True. Default = 128")
    parser.add_argument('--patch_stride', dest="stride", type=int, default=32,
                        help="Steps taken during the patch extraction.")
    parser.add_argument('--min_filter', dest="min_filter", type=int, default=None,
                        help="Minimum filter value")
    parser.add_argument('--model_name', dest="model_name", type=str, default="UNET_",
                        help="The name to save the model with. Epochs will be added automatically. Give the name without the .h5 extension.")
    parser.add_argument('--scan_ext', dest="scan_ext", type=str, default="T1.nii.gz",
                        help="The extension used for identifying the scan file. Default: T1.nii.gz")
    parser.add_argument('--les_ext', dest="les_ext", type=str, default="LESION.nii.gz",
                        help="The extension used for identifying the lesion file. Default: LESION.nii.gz")

    try:
        args = parser.parse_args()
        for i, arg in enumerate(vars(args)):
            print('{}.{}: {}'.format(i, arg, vars(args)[arg]))

        return args

    except:
        parser.print_help()


def get_optimizer(opt, lr):
    # Optimizer Definition
    if opt == 'adam':
        return Adam(learning_rate=lr)
    elif opt == 'sgd':
        return SGD(learning_rate=lr)
    elif opt == 'rmsprop':
        return RMSprop(learning_rate=lr)
    elif opt == 'adadelta':
        return Adadelta(learning_rate=lr)
    elif opt == 'adamax':
        return Adamax(learning_rate=lr)
    elif opt == 'adagrad':
        return Adagrad(learnitng_rate=lr)
    else:
        print("Invalid Optimizer, using Adam as default optimizer.")
        return Adam(learning_rate=lr)


def train(args):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    # Clear the default TensorFlow graph and release resources
    tf.keras.backend.clear_session()

    images = Utils.get_all_possible_files_paths(args.data_path, args.scan_ext)
    masks = Utils.get_all_possible_files_paths(args.data_path, args.les_ext)

    filters = [64, 128, 256, 512, 1024]

    if args.min_filter is not None:
        filters.clear()
        val = args.min_filter
        for i in range(0, 5):
            filters.append(val)
            val = val * 2

    # batch_size = 4  # The batch size to use when training the model
    if args.aply_patch:
        image_shape = (args.patch_size, args.patch_size, args.patch_size)
    else:
        image_shape = (args.image_size, args.image_size, args.image_size)  # The size of the images

    if args.classes == 1:
        loss = 'binary_crossentropy'
    else:
        loss = 'categorical_crossentropy'

    strategy = tf.distribute.MirroredStrategy()

    if args.use_kfold:
        print(f"Applying K-Folds. The training data will be divided into {args.folds}-splits")

        all_losses = []
        all_accuracies = []
        all_dice_scores = []

        kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

        for idx, (train_idx, test_idx) in enumerate(kf.split(images)):
            print(f"Fold {idx} of {args.folds}:")

            # Split the data into training and validation sets for this fold

            x_train, x_test = np.array(images)[train_idx.astype(int)], np.array(images)[test_idx.astype(int)]
            y_train, y_test = np.array(masks)[train_idx.astype(int)], np.array(masks)[test_idx.astype(int)]

            print(f"Training-Instances: {len(x_train)}, Testing-Instances: {len(x_test)}")
            with strategy.scope():
                train_generator = NiiDataGenerator(x_train, y_train, args.batch_size, args.image_size, args.aply_norm,
                                                   args.aply_patch,
                                                   args.patch_size, args.patch_stride)

                if os.path.exists(args.model_save):
                    print("Resuming training from checkpoint:", args.model_save)
                    model = tf.keras.models.load_model(args.model_save, custom_objects={'dice_coef': dice_coef})

                else:
                    model = unet_3d(input_size=(*image_shape, 1), num_classes=args.classes, dropout=args.dropout,
                                    use_maxpool=args.use_maxpool, use_transpose=args.use_transpose,
                                    masks=filters).generate_model()

                # Because of a larger batch_size, we also want to take bigger steps in the direction
                # of the minimum to preserve the number of epochs to converge. Hence, learning rate is multiplied by number of GPUs.
                model.compile(optimizer=get_optimizer(args.optimizer, args.lr * len(gpus)), loss=loss,
                              metrics=['acc', dice_coef])
            checkpoint = ModelCheckpoint(args.model_save.split('.h5')[0] + f"_{idx}" + ".h5", monitor='loss',
                                         save_best_only=True)

            if args.aply_early_stp:
                history = model.fit(train_generator, steps_per_epoch=len(train_generator),
                                    epochs=args.epochs, callbacks=[get_early_stopping(), checkpoint])
            else:
                history = model.fit(train_generator, steps_per_epoch=len(train_generator),
                                    epochs=args.epochs, callbacks=[checkpoint])

            all_losses.append(history.history['loss'])
            all_accuracies.append(history.history['acc'])
            all_dice_scores.append(history.history['dice_coef'])

            # evaluate_test_data(x_test, y_test, f"model_fold_{idx}.h5")

        show_history_kfolds(all_losses, "loss")
        show_history_kfolds(all_accuracies, "acc")
        show_history_kfolds(all_dice_scores, "dice_coef")

    else:
        print("Starting processing without K-Folds.")
        print("Loading Data.")

        if args.test_ratio > 0.0:
            x_train, x_test, y_train, y_test = train_test_split(images, masks, test_size=args.test_ratio,
                                                                random_state=args.seed)
        else:
            x_train, y_train = images, masks
            x_test, y_test = None, None
            if args.test_path is not None:
                x_test = Utils.get_all_possible_files_paths(args.test_path, "T1.nii.gz")
                y_test = Utils.get_all_possible_files_paths(args.test_path, "LESION.nii.gz")

        print("Total-Instances: " + str(len(images)))
        print("Training-Instances: " + str(len(x_train)))

        with strategy.scope():
            if args.valid_ratio != 0.0:
                print("Since, valid ratio != 0.0. Defining valid set. Validation-Instances: "
                      + str(np.ceil(len(x_train) * args.valid_ratio)))

                valid_generator = NiiDataGenerator(x_train[:int(np.ceil(len(x_train) * args.valid_ratio))],
                                                   y_train[:int(np.ceil(len(x_train) * args.valid_ratio))],
                                                   args.batch_size, args.image_size, args.aply_norm, args.aply_patch,
                                                   args.patch_size, args.stride)
                train_generator = NiiDataGenerator(x_train[int(np.ceil(len(x_train) * args.valid_ratio)):],
                                                   y_train[int(np.ceil(len(x_train) * args.valid_ratio)):],
                                                   args.batch_size, args.image_size, args.aply_norm, args.aply_patch,
                                                   args.patch_size, args.stride)
            else:
                train_generator = NiiDataGenerator(x_train, y_train, args.batch_size, args.image_size, args.aply_norm,
                                                   args.aply_patch,
                                                   args.patch_size, args.stride)

            print("Starting Training.")
            if os.path.exists(args.model_save):
                print("Resuming training from checkpoint:", args.model_save)
                model = tf.keras.models.load_model(args.model_save, custom_objects={'dice_coef': dice_coef})
            else:
                model = unet_3d(input_size=(*image_shape, 1), num_classes=args.classes, dropout=args.dropout,
                                use_maxpool=args.use_maxpool, use_transpose=args.use_transpose,
                                masks=filters).generate_model()

            lr_scheduler = get_learning_rate_scheduler(args.lr * len(gpus), decay_steps=10000, decay_rate=0.9)
            model.compile(optimizer=Adam(learning_rate=lr_scheduler), loss=loss, metrics=[dice_coef])

        if valid_generator is None:
            checkpoint = ModelCheckpoint(args.model_name + "{epoch:02d}.h5", monitor='loss', save_best_only=True,
                                         save_freq='epoch')
        else:
            checkpoint = ModelCheckpoint(args.model_name + "{epoch:02d}.h5", monitor='val_loss', save_best_only=True,
                                         save_freq='epoch')

        if args.aply_early_stp:
            history = model.fit(train_generator, validation_data=valid_generator, steps_per_epoch=len(train_generator),
                                epochs=args.epochs, callbacks=[get_early_stopping(), checkpoint])
        else:
            history = model.fit(train_generator, validation_data=valid_generator, steps_per_epoch=len(train_generator),
                                epochs=args.epochs, callbacks=[checkpoint])

        print("Training History")

        if valid_generator is None:
            show_history(history, False)
        else:
            show_history(history, True)

        if x_test is not None and y_test is not None:
            last_save_cp = args.model_name + "{epoch:" + args.epochs + "}.h5"
            if args.aply_patch:
                evaluate_patched_test_data(x_test, y_test, last_save_cp, args.image_size, args.aply_norm,
                                           args.patch_size, args.stride)
            else:
                evaluate_test_data(x_test, y_test, last_save_cp, args.image_size, args.aply_norm)

        else:
            print("Test will be skipped as test_path was not provided.")


if __name__ == "__main__":
    print("Starting Processing")
    args = get_arguments()
    train(args)
