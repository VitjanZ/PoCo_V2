import numpy as np
from cv2 import connectedComponentsWithStats, dilate

def non_max_suppression_reverse(msk, filter_size):
    orig_mask = msk.copy()
    kernel = np.ones((filter_size,filter_size), dtype=np.uint8)
    kernel[filter_size//2, filter_size//2] = 0
    dilated_mask = dilate(orig_mask, kernel, iterations=1)
    local_max_mask = (dilated_mask < orig_mask).astype(np.uint8)
    local_plateau_mask = (dilated_mask == orig_mask).astype(np.uint8)
    local_plateau_mask[dilated_mask == 0] = 0

    _, _, _, centroids = connectedComponentsWithStats(local_plateau_mask)
    centroids = centroids[~np.isnan(centroids).any(axis=1)]
    centroids = centroids.astype(int)
    ret_mask = np.zeros((orig_mask.shape), dtype=orig_mask.dtype)

    ret_mask[centroids[:, 1], centroids[:, 0]] = 1
    ret_mask[local_max_mask > 0] = 1

    return ret_mask
