import ants
import os
import SimpleITK as sitk
import nibabel as nib
import numpy as np
import pandas as pd

from Process.ProcessUtils import save_img
from Utils.CommonUtils import get_all_possible_files_paths


def intensity_clipping_si(src):
    img = nib.load(src).get_fdata()
    lower_percentile = np.percentile(img, 0.5)
    upper_percentile = np.percentile(img, 99.5)

    return np.clip(img, lower_percentile, upper_percentile)


def intensity_clipping(src_path, ext: str = "_SS.nii.gz"):
    for i in get_all_possible_files_paths(src_path, ext):
        print(f"In process: {i}")
        if not os.path.exists(i.split('.nii.gz')[0] + "_IC.nii.gz"):
            img_hdr = nib.load(i)  # image header
            img = img_hdr.get_fdata()
            lower_percentile = np.percentile(img, 0.5)
            upper_percentile = np.percentile(img, 99.5)

            clipped_img = np.clip(img, lower_percentile, upper_percentile)

            ni = nib.Nifti1Image(clipped_img, affine=img_hdr.affine, header=img_hdr.header)
            nib.save(ni, i.split('.nii.gz')[0] + "_IC.nii.gz")

        else:
            print("Skipping processing as it already exists.")


def test_case_1():
    intensity_clipping(
        "/home/mmal151/resmed202200021-IMPRESS_data/Mishaim/ISLES-2024/isles24_batch_1/raw_data/sub-stroke0071/ses-01/sub-stroke0071_ses-01_ncct.nii.gz")
    img_hdr = nib.load(
        "/home/mmal151/resmed202200021-IMPRESS_data/Mishaim/ISLES-2024/isles24_batch_1/raw_data/sub-stroke0071/ses-01/sub-stroke0071_ses-01_ncct.nii.gz")
    ct_img = img_hdr.get_fdata()
    # clipped_img = processing_ct("/home/mmal151/resmed202200021-IMPRESS_data/Mishaim/ISLES-2024/isles24_batch_1/raw_data/sub-stroke0003/ses-01/sub-stroke0003_ses-01_ncct.nii.gz")
    clipped_img = np.clip(ct_img, 0, 80)
    # mri_img = nib.load("/home/mmal151/Desktop/Dummy_Data/T1_1/sub-r001s019/sub-r001s019_T1.nii.gz").get_fdata()

    save_img(clipped_img, '/home/mmal151/Desktop/Dummy_Data/CT/sub-stroke0071_tempp.nii.gz',
             img_hdr)


def get_nifti_info(file_path):
    nifti_img = nib.load(file_path)
    voxel_sizes = nifti_img.header.get_zooms()
    dimensions = nifti_img.shape
    affine_matrix = nifti_img.affine

    return {
        'FilePath': file_path,
        'VoxelSizes': voxel_sizes,
        'Dimensions': dimensions,
        'AffineMatrix': affine_matrix.tolist()
    }


def save_nifti_info(input_path="/home/mmal151/Desktop/Dummy_Data/INPUT", ext="T1.nii.gz"):
    info = []
    for i in get_all_possible_files_paths(input_path, ext):
        print(f"Processing {i}")
        info.append(get_nifti_info(i))

    df_resampled_info = pd.DataFrame(info)
    df_resampled_info.to_csv('/home/mmal151/Desktop/Dummy_Data/nifti_info_sr.csv', index=False)


def resampling(src_path, ext: str = "_SS.nii.gz"):
    for i in get_all_possible_files_paths(src_path, ext):
        print(f"In process: {i}")
        if not os.path.exists(i.split('.nii.gz')[0] + "_SR.nii.gz"):
            img = ants.image_read(i)
            target_spacing = (1, 1, 1)
            resampled_image = ants.resample_image(img, target_spacing, False, 1)
            ants.image_write(resampled_image, i.split('.nii.gz')[0] + "_RS.nii.gz")
        else:
            print("Skipping processing as it already exists.")


if __name__ == "__main__":
    resampling("/home/mmal151/resmed202200021-IMPRESS_data/Mishaim/ISLES-2024_Cleaned", ext="_lesion-msk.nii.gz")
    print("Pause")
