import nibabel as nib
import pandas as pd
import logging

from sklearn.model_selection import train_test_split

from Util.Preprocessing import data_augmentation
from Util.Utils import get_configurations, get_all_possible_files_paths, is_valid_file, str_to_list, \
    get_all_possible_subdirs, remove_dirs

CLASS_NAME = "[DatasetLoader/BalancedLoader]"


# Data is divided into different categories (bin_id) on the basis of the number of voxels in the lesion mask.
#
# Augmentation should also be performed on each bin separately, ensuring that the augmented sample point does not
# exceed the bin range.

def generate_csv(root, x_ext, y_ext, generate_bins=True, bin_range=None):
    if bin_range is None:
        bin_range = [0, 100, 1000, 5000, 10000, 100000]

    X = get_all_possible_files_paths(root, x_ext)
    Y = get_all_possible_files_paths(root, y_ext)

    assert len(X) == len(Y), f"Invalid number of datapoints. Number of input images [{len(X)}] doesn't match number of " \
                             f"masks [{len(Y)}]."

    data = []
    for i in range(0, len(X)):
        mask = nib.load(Y[i])
        voxel = nib.imagestats.count_nonzero_voxels(mask)
        bin_id = None

        if generate_bins:
            for j in range(0, len(bin_range)):
                if (j + 1 < len(bin_range)) and bin_range[j] < voxel < bin_range[j + 1]:
                    bin_id = j
            if bin_id is None:
                bin_id = len(bin_range) - 1

        data.append({'Serial_No': i, 'Img_Path': X[i], 'Mask_Path': Y[i], 'Lesion_Voxel_Vol': voxel, 'Bin_Id': bin_id})

    pd.DataFrame(data).to_csv("data.csv")


def gen_balanced_train_test_split(in_file, test_ratio=0.2, total_bins=6,
                                  train_file="train_set.csv", test_file="test_set.csv",
                                  valid_filename="valid_set.csv", valid_ratio=0.2):
    df = pd.read_csv(in_file)
    bin_masks = [df['Bin_Id'].isin([i]) for i in range(0, total_bins)]
    create = True

    for i in bin_masks:
        if not df[i].empty:
            df_train, df_test = train_test_split(df[i], test_size=test_ratio)
            df_train, df_valid = train_test_split(df_train, test_size=valid_ratio)
            if not (df_train.empty and df_test.empty):
                if create:
                    df_train.to_csv(train_file, index=False)
                    df_test.to_csv(test_file, index=False)
                    df_valid.to_csv(valid_filename, index=False)
                    create = False
                else:
                    df_train.to_csv(train_file, mode='a', index=False, header=False)
                    df_test.to_csv(test_file, mode='a', index=False, header=False)
                    df_valid.to_csv(valid_filename, mode='a', index=False, header=False)


def generate_sets(cfg, ldr_cfg):
    lgr = CLASS_NAME + "[generate_sets()]"
    if not is_valid_file("data.csv"):
        logging.info(f"{lgr}: Data file doesn't exist; generating data")
        generate_csv(cfg["train"]["data"]["inputs"], cfg["train"]["data"]["img_ext"], cfg["train"]["data"]["lbl_ext"],
                     True, str_to_list(ldr_cfg["balanced"]["voxel_range"]))

    gen_balanced_train_test_split("data.csv", cfg["train"]["data"]["test"]["ratio"],
                                  len(ldr_cfg["balanced"]["voxel_range"].split(",")),
                                  ldr_cfg["balanced"]["train_set_file"], ldr_cfg["balanced"]["test_set_file"],
                                  ldr_cfg["balanced"]["valid_set_file"],
                                  cfg["train"]["data"]["valid"]["ratio"])


def load_dataset(cfg):
    lgr = CLASS_NAME + "[load_dataset()]"
    ldr_cfg = get_configurations("config_dataloader.yml")

    if cfg["train"]["data"]["rem_pre_aug"]:
        logging.info(f"{lgr}: Removing previous augmentations.")
        _ = remove_dirs(get_all_possible_subdirs(cfg["train"]["data"]["inputs"], "full_path"), "_cm")

    if ldr_cfg["balanced"]["generate_sets"]:
        generate_sets(cfg, ldr_cfg)

    train_set = pd.read_csv(ldr_cfg["balanced"]["train_set_file"])
    test_set = pd.read_csv(ldr_cfg["balanced"]["test_set_file"])
    valid_set = pd.read_csv(ldr_cfg["balanced"]["valid_set_file"])

    x, y = train_set["Img_Path"].tolist(), train_set["Mask_Path"].tolist()

    if cfg["train"]["data"]["augmentation"]["alw_aug"]:
        x, y = data_augmentation(cfg, x, y, cfg["data"]["inputs"])

    return x, y, valid_set["Img_Path"].tolist(), valid_set["Mask_Path"].tolist(), \
           test_set["Img_Path"].tolist(), test_set["Mask_Path"].tolist()
