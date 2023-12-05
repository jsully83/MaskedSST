import os

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "4"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "4"  # export NUMEXPR_NUM_THREADS=6
os.environ["GDAL_NUM_THREADS"] = "4"

import random
import glob
import pickle
from tqdm import tqdm
import numpy as np
import rasterio as rio
import warnings

import torch
from torch.utils.data import Dataset

wc_labels = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare/sparse vegetation",
    70: "Snow and Ice",
    80: "Permanent water bodies",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen",
}
wc_labels_inv = {v: k for k,v in wc_labels.items()}

wc_labels_train = {
    0: "Tree cover",
    1: "Shrubland",
    2: "Grassland",
    3: "Cropland",
    4: "Built-up",
    5: "Bare/sparse vegetation",
    6: "Snow and Ice",
    7: "Permanent water bodies",
    8: "Herbaceous wetland",
    9: "Mangroves",
    10: "Moss and lichen",
}

dfc_labels = {
    1: "Forest",
    2: "Shrubland",
    3: "Savanna",
    4: "Grassland",
    5: "Wetlands",
    6: "Croplands",
    7: "Urban/Built-up",
    8: "Snow/Ice",
    9: "Barren",
    10: "Water",
}

dfc_labels_inv = {v: k for k,v in dfc_labels.items()}

dfc_labels_train = {
    0: "Forest",
    1: "Shrubland",
    2: "Grassland",
    3: "Wetlands",
    4: "Croplands",
    5: "Urban/Built-up",
    6: "Barren",
    7: "Water",
    -1: "Invalid",
}

wavelengths = [
    418.24 ,  423.874,  429.294,  434.528,  439.603,  444.549,
        449.391,  454.159,  458.884,  463.584,  468.265,  472.934,
        477.599,  482.265,  486.941,  491.633,  496.349,  501.094,
        505.87 ,  510.678,  515.519,  520.397,  525.313,  530.268,
        535.265,  540.305,  545.391,  550.525,  555.71 ,  560.947,
        566.239,  571.587,  576.995,  582.464,  587.997,  593.596,
        599.267,  605.011,  610.833,  616.737,  622.732,  628.797,
        634.919,  641.1  ,  647.341,  653.643,  660.007,  666.435,
        672.927,  679.485,  686.11 ,  692.804,  699.567,  706.401,
        713.307,  720.282,  727.324,  734.431,  741.601,  748.833,
        756.124,  763.472,  770.876,  778.333,  785.843,  793.402,
        801.01 ,  808.665,  816.367,  824.112,  831.901,  839.731,
        847.601,  855.509,  863.455,  871.433,  879.442,  887.478,
        895.537,  902.257,  903.617,  911.715,  911.872,  919.827,
        921.624,  927.951,  931.512,  936.082,  941.53 ,  944.217,
        951.677,  952.355,  960.495,  961.948,  968.638,  972.341,
        976.783,  982.851,  984.932,  993.083,  993.475, 1004.21 ,
       1015.05 , 1026.   , 1037.05 , 1048.19 , 1059.42 , 1070.74 ,
       1082.14 , 1093.62 , 1105.17 , 1116.79 , 1128.47 , 1140.2  ,
       1151.98 , 1163.81 , 1175.67 , 1187.56 , 1199.48 , 1211.42 ,
       1223.37 , 1235.34 , 1247.31 , 1259.3  , 1271.29 , 1283.29 ,
       1295.28 , 1307.27 , 1319.25 , 1331.22 , 1343.18 , 1355.13 ,
       1367.06 , 1378.96 , 1390.84 , 1461.46 , 1473.1  , 1484.69 ,
       1496.24 , 1507.75 , 1519.22 , 1530.64 , 1542.02 , 1553.36 ,
       1564.65 , 1575.9  , 1587.1  , 1598.26 , 1609.36 , 1620.43 ,
       1631.44 , 1642.41 , 1653.33 , 1664.2  , 1675.03 , 1685.8  ,
       1696.53 , 1707.2  , 1717.83 , 1728.4  , 1738.93 , 1749.4  ,
       1759.83 , 1939.44 , 1948.98 , 1958.49 , 1967.95 , 1977.37 ,
       1986.74 , 1996.07 , 2005.36 , 2014.61 , 2023.82 , 2032.99 ,
       2042.11 , 2051.19 , 2060.24 , 2069.24 , 2078.21 , 2087.13 ,
       2096.01 , 2104.86 , 2113.67 , 2122.44 , 2131.17 , 2139.87 ,
       2148.52 , 2157.15 , 2165.73 , 2174.28 , 2182.79 , 2191.27 ,
       2199.71 , 2208.12 , 2216.5  , 2224.84 , 2233.14 , 2241.42 ,
       2249.66 , 2257.86 , 2266.04 , 2274.18 , 2282.29 , 2290.37 ,
       2298.42 , 2306.44 , 2314.42 , 2322.37 , 2330.29 , 2338.19 ,
       2346.05 , 2353.88 , 2361.68 , 2369.45 , 2377.19 , 2384.9  ,
       2392.58 , 2400.23 , 2407.85 , 2415.45 , 2423.01 , 2430.55 ,
       2438.05 , 2445.53,
]
# empty L2 bands
invalid_l2_bands = [
    False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False,
    True, True, True, True, True, True, True, True, True, True, True,
    True, True, True, True, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False,
    False, False, False, False, True, True, True, True, True, True,
    True, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False,
    False, False, False, False,
]

class EnMAPWorldCoverDataset(Dataset):
    def __init__(self, path, img_transforms, label_transform, device, pixel_location_file=None, num_samples_per_class=None, patch_size=3, patch_offset=100, test=False, load_to_memory=False, target_type="worldcover", remove_bands=[], shuffle_samples=False, clip=(-200,10000), rgb_only=False): 
        super().__init__()
        """ if  pixel_location_file is not none, num_samples_per_class pixels will be read in order from the pixel_location_file
        """
        assert target_type in ["worldcover", "dfc", "unlabeled"], f"target_type needs to be either worldcover or dfc: {target_type=}"
        # self.nodata = -32768
        self.nodata = 0.00000001
        # self.invalid_band_idxs = [126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138,
        # 139, 140, 160, 161, 162, 163, 164, 165, 166]
        self.invalid_band_idxs = []
        if remove_bands:
            self.invalid_band_idxs.extend(remove_bands)
        self.path = path
        self.transforms = img_transforms
        self.label_transform = label_transform
        self.device = device
        self.load_to_memory = load_to_memory
        self.testset = test
        self.target_type = target_type
        self.num_samples_per_class = num_samples_per_class
        self.pixel_location_file = pixel_location_file
        self.patch_size = patch_size
        self.patch_offset = patch_offset # only start reading patches lower than that value in the file for each class
        self.shuffle_samples = shuffle_samples # shuffle samples in the list
        self.clip = clip
        self.rgb_only = rgb_only

        if self.pixel_location_file is not None:
            assert 0 < num_samples_per_class < 6172 # max for dfc MexicoCity
            with open(self.pixel_location_file, "rb") as handle:
                self.pixel_locations = pickle.load(handle)

            if self.shuffle_samples:
                for k in list(self.pixel_locations.keys()):
                 random.shuffle(self.pixel_locations[k])

            # all pixels remove beyond the required number
            for k in list(self.pixel_locations.keys()):
                locs = []
                while len(locs) != self.num_samples_per_class:
                    tup = self.pixel_locations[k].pop(self.patch_offset)
                    # don't use pixels at the border of the tile (to make sure they can be patchified later)
                    x,y = tup[1]
                    if (x > self.patch_size):
                        if (x < (64 - self.patch_size)):
                            if (y > self.patch_size):
                                if (y < (64 - self.patch_size)):
                                    locs.append(tup)
                self.pixel_locations[k] = locs

            self.load_from_pixel_location_file_to_memory()

            # for k,v in self.pixel_locations.items():
                # print(k, len(v), len(set([x[0] for x in v])), sorted(set([x[0] for x in v])))

            print(f"{self.pixel_location_file=}")
            print(f"Number of samples: {len(self.patches):,}")

        if self.testset:
            assert "test" in path
        else:
            assert "train" in path

        if self.target_type in ["worldcover", "unlabeled"]:
            print("loading")
            self.enmap_files = glob.glob(os.path.join(path, "*", "*enmap.tif"))
            print(self.enmap_files)
        elif self.target_type == "dfc":
            self.enmap_files = glob.glob(os.path.join(path, "*",  "*enmap.tif"))

        self.target_files = [f.replace("enmap.tif", f"{target_type}_30m.tif") for f in self.enmap_files]

        if self.target_type == "unlabeled":
            self.target_files = None

        if self.pixel_location_file is None:
            print(f"{self.pixel_location_file=}")
            print(f"Number of tiles: {len(self.enmap_files):,}")

        if self.load_to_memory:
            self.imgs = self.load_imgs()
            self.labels = None if self.target_type == "unlabeled" else self.load_labels()

    def __len__(self):
        if self.pixel_location_file is None:
            return len(self.enmap_files)
        else:
            return len(self.patches)

    def load_imgs(self):
        imgs = []
        print("Loading imgs...")
        for idx in tqdm(range(len(self)), total=len(self)):
            img = self.load_img(self.enmap_files[idx])
            imgs.append(img)

        return torch.stack(imgs) 

    def load_img(self, path):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Dataset has no geotransform, gcps, or rpcs. The identity matrix be returned.")
            with rio.open(path, num_threads=4) as f:
                img = f.read([x for x in f.indexes if x-1 not in self.invalid_band_idxs])
        img = self.transforms(img)
        if self.rgb_only:
            img = img[[199, 150, 0]]
        return img

    def load_label(self, path):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Dataset has no geotransform, gcps, or rpcs. The identity matrix be returned.")
            with rio.open(path, num_threads=4) as f:
                label = f.read()[0]
        label = self.label_transform(label)
        return label

    def load_labels(self):
        labels = []
        print("Loading labels...")
        for idx in tqdm(range(len(self)), total=len(self)):
            label = self.load_label(self.target_files[idx])
            labels.append(label)

        return torch.stack(labels) 

    def load_from_pixel_location_file_to_memory(self):
        self.patches = []
        self.labels = []

        k = 0
        prev_file = ""
        for c, v in tqdm(self.pixel_locations.items()):
            for pixel_info in v:
                if pixel_info[0] != prev_file:
                    # load the new tif file, if necessary
                    img = self.load_img(pixel_info[0])
                x,y = pixel_info[1]

                self.patches.append(img[:, x-self.patch_size//2 : (x+self.patch_size//2)+1, y-self.patch_size//2 : (y+self.patch_size//2)+1])
                self.labels.append(c)
                prev_file = pixel_info[0]
                k += 1

    def _getitem_from_list(self, idx):
            img = self.patches[idx]
            label = self.labels[idx] if self.target_type != "unlabeled" else None

            if self.clip is not None:
                img = torch.clip(img, min=self.clip[0], max=self.clip[1])

            return {"img": img, "label": label, "idx": idx}

    def __getitem__(self, idx):

        if self.pixel_location_file is not None:
            return self._getitem_from_list(idx)

        sample = {"idx": idx}

        img = self.imgs[idx] if self.load_to_memory else self.load_img(self.enmap_files[idx])
        if self.clip is not None:
            img = torch.clip(img, min=self.clip[0], max=self.clip[1])
        sample["img"] = img

        if self.target_type != "unlabeled":
            sample["label"] = self.labels[idx] if self.load_to_memory else self.load_label(self.target_files[idx]) 

        return sample

class StandardizeEnMAP(object):
    def __init__(self, use_clipped=True):
        super().__init__()
        # band-wise mean/std values from enmap dataset with 5000 samples
        # note that no-data bands have been removed
        self.use_clipped = use_clipped # clipped at -200:10000

        self.means =  np.array([15488.01822631, 14983.72834872, 15527.21885723, 12722.9852776 ,
                                11673.3922261 , 12048.30101515, 13507.63254849, 15574.81136238,
                                16665.25738682, 17591.48431664, 17800.56573429, 18010.33869397,
                                18129.55486901, 18222.90318821, 18256.56434553, 18168.34499737,
                                18027.7278307 , 17793.41835838, 17279.22252307, 15648.16681988,
                                12428.98764736, 10614.56237611, 11497.39411559, 11983.43859604,
                                14735.31148353, 16136.82734834, 16331.72705513, 16642.44648149,
                                16890.91792984, 17340.33010067, 17744.6234619 , 17867.65373269,
                                17642.89242979, 17508.83079337, 17730.54317482, 17425.23201728,
                                16699.65654785, 15620.48586782, 13638.67235717, 12183.06188   ,
                                9264.48633904, 11477.15340887,  9876.32066317, 10203.09647569,
                                12292.34927129, 12318.54646413, 12507.34247127, 13553.96037928,
                                13929.120096  , 14103.50247628, 14288.88696653, 14457.52949812,
                                14704.74481506, 14867.44786618, 14928.33398124, 15163.26567276,
                                15320.67595836, 15372.53237296, 15464.85503393, 15526.44746804,
                                15599.50816727, 15659.20315452, 15672.96544521, 15685.31524293,
                                15685.31843676, 15605.8029314 , 15538.19582511, 15489.01372717,
                                15433.97764801, 15382.74809604, 15346.3927914 , 15284.78003135,
                                10271.55195983, 11448.29824689, 11455.12923398, 12230.88819009,
                                11751.34208785, 11744.89371538, 12018.7536921 , 12882.06224701,
                                12494.76639711, 12060.67105589, 11743.5044459 , 11856.76843225,
                                12022.36851884, 12156.87144268, 12194.34302876, 12187.78096199,
                                12275.13226021, 12535.54964406, 12644.83973266, 12742.24155454,
                                12976.61715292, 13031.05430793, 13189.91203542, 13306.29810059,
                                13423.48009354, 13491.36510605, 13542.58823588, 13759.85877461,
                                13713.91339381, 13681.86912003, 13666.79051956, 13720.70567948,
                                13763.9524606 , 13723.21312618, 13577.45269115, 13496.02571844,
                                13380.8268252 , 13250.31994906, 13153.71378465, 13066.76417574,
                                12967.38397221, 12889.05107434, 12730.11092424, 12728.62225197,
                                12613.35606638, 12487.2621973 , 12438.32019172, 12144.05231993])

        self.stds = np.array([12735.531657647281, 12721.213270250933, 12753.622014726438, 
                                12713.045660895146, 12820.04377085102, 12828.439477134558, 12753.689392062483, 
                                12852.768293075686, 12862.643350903252, 12728.560551622215, 12766.135386885364, 
                                12811.023796093074, 12832.23712392625, 12840.787167188624, 12845.44025431407, 
                                12817.00647273513, 12794.747334476615, 12781.935260862256, 12870.886609586329, 
                                12900.796971366939, 12789.318284451489, 12927.860035846772, 12937.368966816044, 
                                12998.04730106508, 12966.91014169791, 13065.31212294035, 13094.344871980396, 
                                13071.250074422045, 13000.986966439028, 12894.36933232532, 12802.426932627137,
                                12791.821415752942, 12872.387527948007, 12915.530798497883, 12855.160056644889,
                                12923.313677365604, 13120.914668239759, 13175.11442290776, 13145.295124276441, 
                                13188.275564571808, 11527.691826454971, 11292.550022635485, 10746.012763541323, 
                                11255.817021204002, 13923.689422547706, 13841.740494355407, 13360.630609824759, 
                                13146.30065512405, 12701.686039976734, 12560.10165781337, 12522.754612976847, 
                                12526.051413694548, 12687.910141500748, 12742.265772921572, 12711.332082137877, 
                                12823.94293036102, 12884.288731670329, 12811.012105306068, 12788.121867786143,
                                12788.272131805456, 12854.074715862303, 12874.895634174642, 12904.550086515617, 
                                12938.071902822203, 12986.07704135046, 13044.940553255725, 13125.676785882297,
                                13216.170356097671, 13319.757937033784, 13410.340698687027, 13484.167733172715,
                                13638.693669275239, 13085.45453236416, 13164.300924940702, 13129.776354481568,
                                12152.661313986082, 11820.82859221426, 11671.52526146022, 11961.678597115715,
                                12270.112104301743, 12319.805856861356, 11749.04057028944, 11322.814174011866, 
                                11589.883553194879, 11866.132069775575, 11952.218037348146, 12026.557882222041, 
                                11903.619712978152, 11934.31443256035, 12034.815667953046, 12079.24024430165, 
                                12195.81237204541, 12369.779078127596, 12346.82036537928, 12462.92787565908, 
                                12661.310825537912, 12770.57833300899, 12860.131241570154, 12927.16330787767, 
                                13207.688211771369, 13111.05129168247, 13247.24746766485, 13134.031823529212, 
                                13122.37554343035, 13156.490797116467, 13132.682839584531, 13185.131236800424,
                                13187.489103596696, 13195.596007460264, 13231.882268365482, 13193.55299008,
                                13152.566320408218, 13204.496983937974, 13237.665915647642, 13093.898634266923,
                                13071.59831235561, 13211.82617198746, 12961.035127434896, 13082.601788017266,
                                12771.029765277774])
    def __call__(self, x):
        if self.use_clipped:
            return (x - self.means[:, np.newaxis, np.newaxis]) / self.stds[:, np.newaxis, np.newaxis]
        return (x - self.means[:, np.newaxis, np.newaxis]) / self.stds[:, np.newaxis, np.newaxis]

    def reverse(self, x):
        if self.use_clipped:
            return x * self.stds[:, np.newaxis, np.newaxis] + self.means[:, np.newaxis, np.newaxis]
            # return x * self.stds_clipped[:, np.newaxis, np.newaxis] + self.means_clipped[:, np.newaxis, np.newaxis] 
        return x * self.stds[:, np.newaxis, np.newaxis] + self.means[:, np.newaxis, np.newaxis]
    
class MaxNormalizeEnMAP(object):
    def __init__(self):
        super().__init__()
        # band-wise max values from enmap dataset with 5000 samples
        # note that no-data bands have been removed
        self.maxs = np.array([ 24266.,  23937.,  23599.,  23322.,  23047.,  22809.,  22578.,
        22365.,  22184.,  22001.,  21836.,  21682.,  21531.,  21380.,
        21238.,  21112.,  20979.,  20844.,  20723.,  20597.,  20463.,
        20348.,  20230.,  20109.,  19993.,  19806.,  19182.,  18686.,
        18380.,  18044.,  17918.,  17787.,  17149.,  16703.,  16927.,
        16712.,  16278.,  16381.,  16596.,  16722.,  16850.,  17179.,
        16878.,  16746.,  17066.,  17025.,  16699.,  16155.,  15692.,
        15896.,  17508.,  18037.,  17740.,  17176.,  17705.,  17594.,
        17460.,  17329.,  17213.,  17081.,  17075.,  17061.,  16897.,
        16890.,  16854.,  16848.,  16808.,  16761.,  16734.,  16692.,
        16668.,  16631.,  16594.,  16532.,  16496.,  16465.,  16521.,
        16446.,  16416.,  16468.,  16404.,  16394.,  16460.,  16411.,
        16413.,  16396.,  16360.,  16285.,  16379.,  16276.,  16377.,
        16308.,  16321.,  16336.,  16349.,  16285.,  16319.,  16240.,
        16287.,  16261.,  15917.,  15567.,  15370.,  15410.,  15326.,
        15558.,  15653.,  15965.,  16001.,  15979.,  16232.,  16222.,
        16189.,  16162.,  16156.,  16124.,  16094.,  16066.,  16045.,
        15812.,  15792.,  15772.,  15754.,  15720.,  15891.,  15707.,
        15669.,  15660.,  15651.,  15637.,  15612.,  15619.,
        15602.,  15587.,  15596.,  15588.,  15577.,  15570.,  15563.,
        15558.,  15560.,  15551.,  15542.,  15534.,  15524., 
        15422.,
        15414.,  15414.,  15414.,  15414.,  15413.,  15410.,  15405.,
        15403.,  15401.,  15399.,  15396.,  15395.,  15393.,  15389.,
        15387.,  15371.,  15369.,  15368.,  15368.,  15366.,  15364.,
        15363.,  15361.,  15360.,  15355.,  15355.,  15354.,  15352.,
        15349.,  15348.,  15346.,  15358.,  15356.,  15355.,  15354.,
        15352.,  15350.,  15353.,  15353.,  15351.,  15350.,  15348.,
        15346.,  15344.,  15339.,  15336.,  15333.,  15331.,  15328.,
        15327.,  15323.,  15322.,  15319.,  15317.,  15314.,  15315.])
    
    def __call__(self, x):
        return x / self.maxs[:, np.newaxis, np.newaxis]

    def reverse(self, x):
        return x * self.maxs[:, np.newaxis, np.newaxis]

class MaxNormalizeAllBandsSame(object):
    def __init__(self):
        super().__init__()
        self.max = np.array([25000.])

    def __call__(self, x):
        return x / self.max

    def reverse(self, x):
        return x * self.max
    
class ToTensor(object):
    def __init__(self):
        super().__init__()
    
    def __call__(self, x):
        return torch.Tensor(x).to(torch.float32)

class WorldCoverLabelTransform(object):
    def __init__(self):
        super().__init__()
        # map labels from 10-100 to 0-10 range 
        self.old_to_new_labels = {
            0: -1,
            10: 0,
            20: 1,
            30: 2,
            40: 3,
            50: 4,
            60: 5,
            70: 6,
            80: 7,
            90: 8,
            95: 9,
            100: 10,
        }
        self.new_to_old_labels = {v:k for k,v in self.old_to_new_labels.items()}
    
    def __call__(self, x):
        x = torch.tensor(x, dtype=torch.long)

        x[x == 100] = 11
        x[x == 90] = 10
        x = torch.div(x, 10, rounding_mode='floor') - 1

        return x

class DFCLabelTransform(object):
    def __init__(self):
        super().__init__()
        # remove unused labels 3,8 and start labels at 0 instead of 1
        self.old_to_new_labels = {
            1: 0,
            2: 1,
            3: -1,
            4: 2,
            5: 3,
            6: 4,
            7: 5,
            8: -1,
            9: 6,
            10: 7,
        }
        self.new_to_old_labels = {v: k for k,v in self.old_to_new_labels.items()}

    def __call__(self, x):
        x = torch.tensor(x, dtype=torch.long)

        x[x == 3] = 0
        x[x == 8] = 0
        x[x >= 3] -= 1
        x[x >= 8] -= 1
        x -= 1

        return x

    def reverse(self, x):
        # add +1 back
        x += 1
        return x

    