# hypso data

import os

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "4"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "4"  # export NUMEXPR_NUM_THREADS=6
os.environ["GDAL_NUM_THREADS"] = "4"

import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio as rio
import torch
import glob
from tqdm import tqdm

labels = {
    0: "Sea", 
    1: "Land", 
    2: "Cloud"
}

classes = list(range(3))

wavelengths = [x/1000.0 for x in range(400000,800000,3333)]

bands = list(range(120))

class HypsoDataset(Dataset):
    def __init__(
        self,
        path,
        label_path,
        transforms=None,
        label_transforms=None,
        patch_size=8,
        test=False,
        rgb_only=False,  
    ):
        super().__init__()
        self.path = path
        self.label_path = label_path
        self.transforms = transforms # result of standardization
        self.label_transforms = label_transforms # doesnt really do anything here but intent is to remove unclassified labels
        self.patch_size = patch_size
        self.rgb_only = rgb_only

        # in `test` mode, return non-overlapping patches sequentially s.t. evetually the whole test set is covered
        # if `test` is false, patches are sampled randomly from the (training) dataset
        self.test = test

        # split of training vs val is handled in utils.py
        self.hypso_files = glob.glob(os.path.join(path, "*", "*radiance.tif"))

        def __len__(self):
            return len(self.hypso_files)
        
        def load_imgs(self):
            imgs = []
            print("Loading imgs...")
            for idx in tqdm(range(len(self)), total=len(self)):
                img = self.load_img(self.enmap_files[idx])
                imgs.append(img)

            return torch.stack(imgs)

class StandardizeHypso(object):
    def __init__(self):
        super().__init__()
    
        self.stds = 

        self.means = [0.0, 0.0, 0.0, 0.0, 63.721881849200315, 67.54847926973412, 70.80471664800739, 71.85024830691319, 71.24505679643555, 70.00910650033346, 67.06848166633957, 61.064942152490254, 66.29818544569507, 69.3291166053067, 69.74759383945286, 72.97264400658332, 73.92865389285768, 76.43634599667548, 74.0838192569872, 74.46365416254845, 74.06725982054361, 72.31010917258442, 70.22968470082962, 69.9797522934285, 70.4800087838226, 70.95804146457704, 71.0133511733881, 66.08698564540016, 65.72816011889388, 66.23489471033886, 66.09909613738847, 64.09824188463188, 62.52665601325293, 62.867800104341036, 62.462117222670024, 61.22235132486981, 58.238388045684786, 58.962266927929385, 60.59895275991553, 60.03130474640527, 60.95168802046201, 60.74909249689442, 60.51788117262136, 59.426066142476074, 59.672658115019566, 59.58841040533365, 59.535372449898475, 59.234471822418605, 57.88199099234541, 57.55517821095484, 57.091807923978905, 55.99899207380163, 55.47235774853064, 55.13597831570616, 54.88818553226556, 55.580822469295995, 54.604675639557804, 51.92798020726902, 52.13189282809286, 52.318955985013496, 52.16475740521177, 52.376708804372214, 52.53714479298515, 52.019375774017924, 51.41006552349257, 51.113397190371415, 51.60690747964943, 50.85633270407148, 49.91163883770975, 49.759486994000575, 50.560436057518764, 51.18703759989584, 51.0333087198768, 50.70656347003184, 49.53336205819175, 49.8731741460878, 48.25248331297989, 49.06050565123112, 50.96448951220852, 51.032468410847606, 50.9697200887467, 50.72950092340104, 50.84805023807254, 50.94491493464186, 50.796005999771644, 46.584620633176826, 44.053224751763615, 47.30340738558694, 49.59540076849495, 50.10949658525362, 51.27452052271473, 53.28732578153532, 54.711057897776726, 55.115785181078095, 48.37437745025324, 46.24291443588363, 49.02134644687096, 48.643740313389216, 51.227300535133494, 57.439363717458, 59.47589301209183, 61.365208802587695, 63.60551647230896, 64.51360739319563, 64.34963339692328, 64.6333333490907, 56.20096610302742, 26.521420011011248, 36.05909481540042, 56.910260087772045, 63.05565745574306, 63.87450113244978, 64.02890224998029, 64.12085537625957, 63.84323256569257, 62.411172873812475, 62.2441364497041, 63.86284402271001, 66.23990014602074, 68.12959688819835]# TODO

    def __call__(self, x):
        return (x - self.means[:, np.newaxis, np.newaxis]) / self.stds[
            :, np.newaxis, np.newaxis
        ]

    def reverse(self, x):
        return (
            x * self.stds[:, np.newaxis, np.newaxis]
            + self.means[:, np.newaxis, np.newaxis]
        )

class HypsoLabelTransform(object):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        # no changes to hypso
        x = torch.tensor(x, dtype=torch.long)
        return x