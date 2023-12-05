import rasterio
import glob
import os
from pathlib import Path
import numpy as np

directory = Path("C:/Users/akuru/Documents/ECEE 7370 Advanced Comp Vision/sea_cloud_land/1-DATA WITH GROUND-TRUTH LABELS")
NUM_BANDS = 120 
IMAGE_SIZE = 956 * 684

# Find all zip files in the specified directory
l2_product_dirs = [
    x
    for x in glob.glob(os.path.join(directory, "*", "DATA"))
    if os.path.isdir(x)
]
l2_spectral_products = [
    # glob.glob(os.path.join(ENMAP_PATH, "*SPECTRAL_IMAGE.TIF"))[0]
    glob.glob(os.path.join(d, "*radiance.tif")) for d in l2_product_dirs
]
num_images = len(l2_spectral_products)
# j=0
# cum_val = [0] * NUM_BANDS
# for folder in l2_spectral_products:
#     with rasterio.open(folder[0]) as src1:
#         # Count the total number of bands
#         total_bands = src1.count
#         # Define metadata for the new dataset
#         merged_meta = src1.meta.copy()
#         merged_meta.update({
#             "count": total_bands
#         })
        
#         # Write bands from the first file
#         for i in range(total_bands):
#             band = src1.read(i+1)
#             cum_val[i] = cum_val[i] + np.sum(band)
            
#     j = j + 1

# print("Mean = ")
# mean = np.array(cum_val) / (IMAGE_SIZE*num_images)
# print(mean.tolist())

stds = []
 
for bands in range(NUM_BANDS):
    for i, geotiff in enumerate(l2_spectral_products):
        with rasterio.open(geotiff[0]) as src1:       
            tensor = np.array(src1.read(bands+1))
            tensor = tensor.reshape([1, -1])
            if bands==0 and i==0:
                all_vals = tensor
            else:
                all_vals = np.hstack([all_vals, tensor])
    print(bands, end = " ")
    stds.append(np.std(all_vals))

print("std = ")
print(stds.tolist())