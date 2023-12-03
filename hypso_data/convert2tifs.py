import rasterio
import numpy as np
import sys
import glob
import os
from sklearn.preprocessing import MinMaxScaler

# This is for converting images to tifs, not the ground truth labels.

SEAMAP_PATH = "C:/Users/akuru/Documents/ECEE 7370 Advanced Comp Vision/sea_cloud_land/1-DATA WITH GROUND-TRUTH LABELS"

products = [
        x
        for x in glob.glob(os.path.join(SEAMAP_PATH, "*", "DATA", "*radiance.npy"))
    ]

IMAGE_SCALE = (0, 65535)
minmax = MinMaxScaler(IMAGE_SCALE)

def scale_vals16(arr):
        """
        Scale the images to 16-bits.
 
        Arguments:
            arr {numpy array} -- images
 
        Returns:
            numpy array -- 16-bit images
        """
        arr_ = arr
        if arr.ndim > 2:
            arr_ = arr_.reshape(-1,arr.shape[2])

        arr_ = minmax.fit_transform(arr_)
        arr_ = arr_.reshape(arr.shape)
        return arr_.astype(np.uint16)

for file in products:
    Z_float = np.load(file)
    Z = scale_vals16(Z_float)
    i = file.index(".npy")
    out_file = file[0:i] + '.tif'

    new_dataset = rasterio.open(
        out_file,
        'w',
        driver='GTiff',
        height=Z.shape[0],
        width=Z.shape[1],
        count=Z.shape[2],
        dtype=Z.dtype
    )
    for i in range(Z.shape[2]):
        img = Z[:, :, i]
        new_dataset.write(img, i+1)
    new_dataset.close()