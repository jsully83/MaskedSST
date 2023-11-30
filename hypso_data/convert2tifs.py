import rasterio
import numpy as np
import sys
import glob
import os

SEAMAP_PATH = "C:/Users/akuru/Documents/ECEE 7370 Advanced Comp Vision/sea_cloud_land/1-DATA WITH GROUND-TRUTH LABELS"

products = [
        x
        for x in glob.glob(os.path.join(SEAMAP_PATH, "*", "DATA", "*radiance.npy"))
        
    ]

for file in products:
    Z = np.load(file)
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