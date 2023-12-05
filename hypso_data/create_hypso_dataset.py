# Cut HYPSO1 L2 tiles into patches and split into train and test set

import os
import glob
import pyproj
import rasterio
from tqdm import tqdm
from rasterio.warp import Resampling

from shapely.ops import transform

TILE_SIZE = 64
UPSCALE_FACTOR_HYPSO = 1  # 3 for 10m resolution with bilinear
type = "labeled" #choices are labeled or unlabeled

HYPSO_PATH = "C:/Users/akuru/Documents/ECEE 7370 Advanced Comp Vision/sea_cloud_land/1-DATA WITH GROUND-TRUTH LABELS"

if type == "labeled":
    OUTPUT_DIR = "C:/Users/akuru/Documents/ECEE 7370 Advanced Comp Vision/sea_cloud_land/gt_tiles"
else:
    OUTPUT_DIR = "C:/Users/akuru/Documents/ECEE 7370 Advanced Comp Vision/sea_cloud_land/tiles"


if __name__ == "__main__":
    wgs84 = pyproj.CRS("EPSG:4326")

    # hypso_products = [
    #     x
    #     for x in glob.glob(os.path.join(HYPSO_PATH, "*", "DATA", "*radiance.tif"))
        
    # ]
    if type == "labeled":
        ending = "*hypso_gt.tif"
        folder = "GROUND-TRUTH LABELS"
    else:
        ending = "*radiance.tif"
        folder = "DATA"

    hypso_products = [
        x
        for x in glob.glob(os.path.join(HYPSO_PATH, "*", folder, ending))
    ]
    
    print(f"Found {len(hypso_products)} HYPSO1 products")

    # make sure that there are no duplicate hypso files
    filenames = [x.split("/")[-1] for x in hypso_products]
    assert len(filenames) == len(set(filenames))
    
    # with open(TESTFILES) as f:
    #     testfiles = [x.strip() for x in f.readlines()]

    for hypso_product in hypso_products:
        if type == "labeled":
            index_str = "\\GROUND-TRUTH LABELS"
            idx = hypso_product.index(index_str)
            filename = hypso_product[idx+21:]
            filename = filename.split("/")[-1].split("-hypso_gt.tif")[0]
        else:
            index_str = "\\DATA"
            idx = hypso_product.index(index_str)
            filename = hypso_product[idx+6:]
            filename = filename.split("/")[-1].split("-radiance.tif")[0]
        
       
        outdir = os.path.join(OUTPUT_DIR, filename)

        # if filename in testfiles:
        #     outdir = outdir.replace("train", "test")

        if os.path.exists(outdir):
            print("Directory already exists/HYPSO1 file already processed")
            save = False
            continue
        else:
            os.mkdir(outdir)
            save = True

        with rasterio.open(hypso_product) as dataset:
            # resample data to target shape
            hypso_meta = dataset.meta.copy()
            hypso_meta["bounds"] = dataset.bounds
            hypso = dataset.read(
                out_shape=(
                    dataset.count,
                    int(dataset.height * UPSCALE_FACTOR_HYPSO),
                    int(dataset.width * UPSCALE_FACTOR_HYPSO),
                ),
                resampling=Resampling.bilinear,
            )

            # scale image transform
            transform = dataset.transform * dataset.transform.scale(
                (dataset.width / hypso.shape[-1]), (dataset.height / hypso.shape[-2])
            )
            hypso_meta["transform"] = transform
            hypso_meta["width"] = hypso.shape[-1]
            hypso_meta["height"] = hypso.shape[-2]

        # create tiles from uncorrupted pixels
        tiles = []
        for i in range(0, hypso.shape[1], TILE_SIZE):
            for j in range(0, hypso.shape[2], TILE_SIZE):
                if i + TILE_SIZE > hypso.shape[1] or j + TILE_SIZE > hypso.shape[2]:
                    continue

                hypso_tile = hypso[:, i : i + TILE_SIZE, j : j + TILE_SIZE]

                if (hypso_tile == hypso_meta["nodata"]).mean(axis=(1, 2)).all():
                    # all bands are nodata for every pixel
                    continue

                tiles.append(hypso_tile)

        print(f"Number of valid tiles for {hypso_product.split('/')[-1]}: {len(tiles)}")

        if save:
            for idx, tile in tqdm(enumerate(tiles), total=len(tiles)):
                if type == "labeled":
                    outstr = f"tile{idx}_hypso_gt.tif"
                else:
                    outstr = f"tile{idx}_hypso.tif"

                no_data = -32768.0 if tile.dtype==int else 10
                with rasterio.open(
                    os.path.join(outdir, outstr),
                    "w",
                    driver="GTiff",
                    nodata=no_data,
                    dtype=tile.dtype,
                    count=tile.shape[0],
                    width=tile.shape[2],
                    height=tile.shape[1],
                ) as f:
                    f.write(tile)
        else:
            print("Not saved, see above")
