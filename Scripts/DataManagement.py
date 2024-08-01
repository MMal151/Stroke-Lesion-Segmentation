import nibabel as nib
import os.path
import shutil

import numpy as np

from Process.ProcessUtils import save_img
from Scripts.MRI_Preprocessing import bias_correction
from Utils.CommonUtils import get_all_possible_files_paths, is_valid_dir


# Deletes all unwarranted files from the repository. Assuming Lesion files end with _LESION.nii.gz and Input Files end with _T1.nii.gz.
# Any files not using this structure will be deleted from the local copy.
def clean_dataset_ATLAS(input_path):
    print("Removing the following files from the input directory.")
    for i in get_all_possible_files_paths(input_path, ""):
        # i.split('/')[-2] would be the id of the patient. Example i.split('/')[-2] = sub-r001s001
        if i.split('/')[-1] != i.split('/')[-2] + "_LESION.nii.gz" \
                and i.split('/')[-1] != i.split('/')[-2] + "_T1.nii.gz" \
                and (i.split('/')[-1] != i.split('/')[-2] + "_T1_bet.nii.gz"
                     and i.split('/')[-1] != i.split('/')[-2] + "_T1_brain.nii.gz"):
            print(i)
            os.remove(i)


def clean_dataset_ISLES(input_path):
    print("Removing the following files from the input directory.")
    for i in get_all_possible_files_paths(input_path, ""):
        # i.split('/')[-2] would be the id of the patient. Example i.split('/')[-2] = sub-r001s001
        if i.split('/')[-1] != i.split('/')[-2] + "_ncct.nii.gz" \
                and i.split('/')[-1] != i.split('/')[-2] + "_lesion-msk.nii.gz" \
                and (i.split('/')[-1] != i.split('/')[-2] + "_ncct_SR.nii.gz"):
            print(i)
            os.remove(i)


# Prepare ATLAS dataset in the same naming and file structure as IMPRESS dataset for smoother transition.
def prepare_atlas(input_path, save_path=""):
    if not is_valid_dir(save_path):
        save_path = input_path + "_Cleaned/"
    else:
        print("Removing previously placed data.")
        shutil.rmtree(save_path)

    print("Preparing ATLAS Dataset")
    for curr_pth in os.listdir(input_path):
        temp_dict = {}
        for file in get_all_possible_files_paths(os.path.join(input_path, curr_pth), ""):
            if file.endswith(".nii.gz"):
                new_save_path = save_path + file.split("ATLAS_2")[1].split("/")[-4] + "/"
                if file.__contains__("T1lesion_mask"):
                    temp_dict["Lesion_Src"] = file
                    temp_dict["Lesion"] = new_save_path + file.split("ATLAS_2")[1].split("/")[-4] + "_LESION.nii.gz"
                elif file.__contains__("Sym_T1w"):
                    temp_dict["T1_Src"] = file
                    temp_dict["T1"] = new_save_path + file.split("ATLAS_2")[1].split("/")[-4] + "_T1.nii.gz"

                if len(temp_dict) >= 4:
                    if not is_valid_dir(new_save_path):
                        os.makedirs(new_save_path, exist_ok=True)
                    shutil.copy(temp_dict["Lesion_Src"], temp_dict["Lesion"])
                    shutil.copy(temp_dict["T1_Src"], temp_dict["T1"])
                    temp_dict.clear()
            else:
                print("Removing file asR it is not a .nii file. File: " + file)


def prepare_ISLES_2024(src, save_path, in_ext, out_ext):
    for i in get_all_possible_files_paths(src, ".nii.gz"):
        if i.__contains__(in_ext) or i.__contains__(out_ext):
            sub_name = save_path + "/" + i.split('/')[-1].split('_')[0]

            if not is_valid_dir(sub_name):
                os.makedirs(sub_name, exist_ok=True)

            shutil.copy(i, sub_name)


def intensity_clipping(src, prefix="_ncct_BC.nii.gz", upper=80, lower=0):
    for i in get_all_possible_files_paths(src, prefix):
        if not os.path.exists(i.split('.nii.gz')[0] + "_IC.nii.gz"):
            print(f"Currently processing {i}")
            img = nib.load(i)
            clipped_img = np.clip(img.get_fdata(), lower, upper)
            nib.save(nib.Nifti1Image(clipped_img, img.affine, img.header), i.split('.nii.gz')[0] + "_IC.nii.gz")
        else:
            print("File already exists.")


if __name__ == "__main__":
    # clean_dataset_ISLES("/home/mmal151/resmed202200021-IMPRESS_data/Mishaim/ISLES-2024_Cleaned/")
    #prepare_ISLES_2024("/home/mmal151/resmed202200021-IMPRESS_data/Mishaim/ISLES-2024/",
    #                   "/home/mmal151/resmed202200021-IMPRESS_data/Mishaim/ISLES-2024_Cleaned/",
     #                  "_ncct.nii.gz", "lesion-msk.nii.gz")

    bias_correction("/home/mmal151/resmed202200021-IMPRESS_data/Mishaim/ISLES-2024_Cleaned/", "ncct.nii.gz")
