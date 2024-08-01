import os
import SimpleITK as sitk

from Utils.CommonUtils import get_all_possible_files_paths


# This file contains all of the functions required to clean-up the ATLAS dataset.


# Performed using HD-BET -- Performs better
# Pre-Req: HD-BET needs to be downloaded and configured alongside the models.
def skull_striping_hd_bet(src_path):
    for i in get_all_possible_files_paths(src_path, "_T1.nii.gz"):
        print(f"In process file: {i}")
        if not os.path.exists(i.split(".nii.gz")[0] + "_bet.nii.gz"):
            command = "hd-bet -i " + i + " -device cpu -mode fast -tta 0"
            os.system(command)
        else:
            print("File already exists.")
    return


# Performed using ITK-Snap
def skull_striping(src_path, frac="0.2"):
    for i in get_all_possible_files_paths(src_path, "_T1.nii.gz"):
        print(f"In process file: {i}")
        if not os.path.exists(i.split(".nii.gz")[0] + "_SS.nii.gz"):
            command = "bet " + i + " " + i.split(".nii.gz")[0] + "_SS.nii.gz" + " -R -f " + str(frac) + " -g 0"
            os.system(command)
        else:
            print("File already exists.")
    return


def bias_correction(src_path, prefix: str = "_SS.nii.gz"):
    for i in get_all_possible_files_paths(src_path, prefix):
        print(f"In process: {i}")
        if not os.path.exists(i.split('.nii.gz')[0] + "_BC.nii.gz"):
            img = sitk.ReadImage(i, sitk.sitkFloat32)
            correctedImg = sitk.N4BiasFieldCorrection(img)
            sitk.WriteImage(correctedImg, i.split('.nii.gz')[0] + "_BC.nii.gz")
        else:
            print("Skipping processing as it already exists.")
