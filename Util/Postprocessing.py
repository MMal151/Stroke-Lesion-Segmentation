import cc3d as cca
import numpy as np


def thresholding(lbl, thresh=0.5):
    return np.ones(lbl.shape) * (lbl > thresh)


# Perform Connected Component Analysis
# Inputs: labels_in -> Input Labels
#         connectivity -> Possible Values: 6, 18, 26 (6 -> Voxel needs to share a face, 18 -> Voxels can share either
#                                                     a face or an edge, 28 -> Common face, edge or corner)
#         k -> k-number of highest connected bodies will be considered in the mask.
def perform_cca(labels_in, connectivity=6, k=3):
    labels_out = cca.largest_k(labels_in, k=k, connectivity=connectivity, delta=0, return_N=False)
    return labels_in * (labels_out > 0)
