import rasterio
import numpy as np
import sys
import glob
import os

SEAMAP_PATH = "C:/Users/akuru/Documents/ECEE 7370 Advanced Comp Vision/sea_cloud_land/1-DATA WITH GROUND-TRUTH LABELS"

products = [
        x
        for x in glob.glob(os.path.join(SEAMAP_PATH, "*", "GROUND-TRUTH LABELS", "*FORMAT.npy"))
    ]


for file in products:
    Z = np.load(file)
    i = file.index("class_NPY_FORMAT.npy")
    
    id = file.index("1-DATA WITH GROUND-TRUTH LABELS")
    id2 = file.index("GROUND-TRUTH LABELS", id+32+1)
    folder = file[id+32:id2-1]
    out_file = file[0:id2+20] + folder + '-hypso_gt.tif'


    new_dataset = rasterio.open(
        out_file,
        'w',
        driver='GTiff',
        height=Z.shape[0],
        width=Z.shape[1],
        count=1,
        dtype=Z.dtype
    )
    
    img = Z[:, :]
    new_dataset.write(img, 1)
    new_dataset.close()