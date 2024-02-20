# -*- coding: utf-8 -*-
"""
Created on Thu May 27 15:17:22 2021

@author: 73239
"""

# Source:  https://github.com/ZhangxinruBIT/CarveMix/blob/main/Task100_ATLASwithCarveMix/Simple_CarveMix.py
import numpy as np
import SimpleITK as sitk
import nibabel as nib
from scipy import ndimage


def get_distance(f, spacing):
    """Return the signed distance."""

    dist_func = ndimage.distance_transform_edt
    distance = np.where(f, -(dist_func(f, sampling=spacing)),
                        dist_func(1 - f, sampling=spacing))

    return distance


def get_head(img_path):
    temp = sitk.ReadImage(img_path)
    spacing = temp.GetSpacing()
    direction = temp.GetDirection()
    origin = temp.GetOrigin()

    return spacing, direction, origin


def copy_head_and_right_xyz(data, spacing, direction, origin):
    TrainData_new = data.astype('float32')
    TrainData_new = TrainData_new.transpose(2, 1, 0)
    TrainData_new = sitk.GetImageFromArray(TrainData_new)
    TrainData_new.SetSpacing(spacing)
    TrainData_new.SetOrigin(origin)
    TrainData_new.SetDirection(direction)

    return TrainData_new


"""
==========================================
The input must be nii.gz which contains 
import header information such as spacing.
Spacing will affect the generation of the
signed distance.
=========================================
"""


def generate_new_sample(image_a, image_b, label_a, label_b):
    spacing, direction, origin = get_head(image_a)

    target_a = nib.load(image_a).get_fdata()
    target_b = nib.load(image_b).get_fdata()
    label_a = nib.load(label_a).get_fdata()
    label_b = nib.load(label_b).get_fdata()
    label = np.copy(label_b)

    dis_array = get_distance(label, spacing)  # creat signed distance
    #     c = np.random.beta(1, 1)#[0,1]             #creat distance
    #     λl = np.min(dis_array)/2                   #λl = -1/2|min(dis_array)|
    #     λu = -np.min(dis_array)                    #λu = |min(dis_array)|
    #     lam = np.random.uniform(λl,λu,1)           #λ ~ U(λl,λu)
    c = np.random.beta(1, 1)  # [0,1] creat distance
    c = (
                    c - 0.25) * 2  # [-1.1] # Original value for 0.25 was 0.5. However with 0.25 the results were better for medium sized lesions
    if c > 0:
        lam = c * np.min(dis_array) / 2  # λl = -1/2|min(dis_array)|
    else:
        lam = c * np.min(dis_array)

    mask = (dis_array < lam).astype('float32')  # creat M

    new_target = target_a * (mask == 0) + target_b * mask
    new_label = label_a * (mask == 0) + label_b * mask

    new_target = copy_head_and_right_xyz(new_target, spacing, direction, origin)
    new_label = copy_head_and_right_xyz(new_label, spacing, direction, origin)
    mask = copy_head_and_right_xyz(mask, spacing, direction, origin)

    return new_target, new_label, mask, lam
