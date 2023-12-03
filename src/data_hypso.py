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
import warnings

labels = {
    0: "Ground",
    2: "Cloud"
}

classes = [list(range(2))] # not sure if this should be [0, 2] instead?

wavelengths = np.linspace(400, 800, num=120).tolist()

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
        target_type="labeled"  
    ):
        super().__init__()
        self.path = path
        self.label_path = label_path
        self.transforms = transforms # result of standardization
        self.label_transforms = label_transforms # doesnt really do anything here but intent is to remove unclassified labels
        self.patch_size = patch_size
        self.rgb_only = rgb_only
        self.target_type = target_type

        # in `test` mode, return non-overlapping patches sequentially s.t. evetually the whole test set is covered
        # if `test` is false, patches are sampled randomly from the (training) dataset
        self.test = test

        # split of training vs val is handled in utils.py
        self.hypso_files = glob.glob(os.path.join(path, "*", "*hypso.tif"))
        tmp = [f.replace("hypso.tif", f"hypso_gt.tif") for f in self.hypso_files]
        tmp = [f.replace("-radiance", f"") for f in tmp]
        self.target_files = [f.replace("tiles", f"gt_tiles") for f in tmp]
        
    def __len__(self):
        return len(self.hypso_files)
    
    def load_imgs(self):
        imgs = []
        print("Loading imgs...")
        for idx in tqdm(range(len(self)), total=len(self)):
            img = self.load_img(self.hypso_files[idx])
            imgs.append(img)

        return torch.stack(imgs)
    
    def load_img(self, path):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Dataset has no geotransform, gcps, or rpcs. The identity matrix be returned.")
            with rio.open(path, num_threads=4) as f:
                img = f.read([x for x in f.indexes])
        img = self.transforms(img)
        if self.rgb_only:
            img = img[[199, 150, 0]]
        return img
    
    def load_label(self, path):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Dataset has no geotransform, gcps, or rpcs. The identity matrix be returned.")
            with rio.open(path, num_threads=4) as f:
                label = f.read()[0]
        label = self.label_transforms(label)
        return label

    def load_labels(self):
        labels = []
        print("Loading labels...")
        for idx in tqdm(range(len(self)), total=len(self)):
            label = self.load_label(self.target_files[idx])
            labels.append(label)

        return torch.stack(labels) 

    def __getitem__(self, idx):
        sample = {"idx": idx}

        img = self.load_img(self.hypso_files[idx])
        # if self.clip is not None:
        #     img = torch.clip(img, min=self.clip[0], max=self.clip[1])
        sample["img"] = img

        if self.target_type != "unlabeled":
            sample["label"] = self.load_label(self.target_files[idx]) 

        return sample

class StandardizeHypso(object):
    def __init__(self):
        super().__init__()

        self.stds = np.array([0.00000001, 0.00000001, 0.00000001, 0.00000001, 14384.226911607719, 14433.78570864593, 14542.411846916471, 14655.77851663927, 14685.04242701548, 14764.58151418035, 14716.662336711472, 14597.558735214116, 14654.70820196955, 14681.45257834262, 14756.741635498212, 14736.72994020985, 14779.318140244255, 14837.821089074585, 14880.866759832705, 15037.058214800336, 15198.05493139463, 15354.011265838117, 15412.469362351854, 15439.339922135272, 15855.798729763677, 15818.686387576927, 16439.486965927623, 16251.994483355427, 15909.554389277304, 16087.34372842837, 16184.00259282304, 16089.893740037181, 16348.471829877055, 16474.12678613427, 16393.98742424684, 16601.093224298274, 15905.948262872775, 15986.905791140842, 16197.57914165936, 16384.693771305407, 16492.258280496593, 16395.332815121143, 16404.09451986886, 16394.44432079088, 16329.90635415218, 16508.336574017958, 16576.204945308844, 16394.921107919232, 16460.03524042387, 16443.500934773598, 16483.22567547224, 16453.187270957347, 16444.6365850474, 16521.64437520549, 16555.13048912102, 16732.764380234228, 16709.181349404025, 16447.465651872582, 16438.681822208568, 16537.04787978957, 16524.252812138107, 16533.94887318293, 16530.807591129684, 16487.94502684679, 16462.530865395656, 16469.15383963927, 16526.40394335226, 16476.657004108092, 16466.649722190334, 16428.123427459526, 16442.409533452115, 16477.50308519431, 16421.131230097086, 16419.48226641547, 16358.872171221285, 16333.740507108056, 16276.14661834973, 16293.51289046671, 16355.02401761492, 16382.06711115385, 16411.828735240717, 16390.689274563578, 16352.286865198937, 16353.17391448929, 16391.19813136667, 16110.614507463126, 15951.874002758206, 16038.170542264224, 16075.892399622693, 16019.016587402657, 15908.845907879453, 15922.675733599888, 16013.613488725883, 16159.493993120694, 15913.56224424974, 16220.106158562532, 16542.28417686677, 16653.42022771785, 16719.75723288943, 16948.04533924122, 17143.629767304654, 17217.732700030727, 17279.232650259943, 17362.363692347117, 17389.96744682854, 17443.125263501086, 16774.910663661034, 15062.264514591878, 16543.782897016477, 17247.488259067257, 17486.13888809634, 17511.938968925522, 17586.95707996008, 17628.288048435836, 17668.92605063691, 17717.04781639687, 17738.93679227333, 17712.251421030956, 17748.682106205742, 17689.242872526527])
        self.means = np.array([0.0, 0.0, 0.0, 0.0, 63.721881849200315, 67.54847926973412, 70.80471664800739, 71.85024830691319, 71.24505679643555, 70.00910650033346, 67.06848166633957, 61.064942152490254, 66.29818544569507, 69.3291166053067, 69.74759383945286, 72.97264400658332, 73.92865389285768, 76.43634599667548, 74.0838192569872, 74.46365416254845, 74.06725982054361, 72.31010917258442, 70.22968470082962, 69.9797522934285, 70.4800087838226, 70.95804146457704, 71.0133511733881, 66.08698564540016, 65.72816011889388, 66.23489471033886, 66.09909613738847, 64.09824188463188, 62.52665601325293, 62.867800104341036, 62.462117222670024, 61.22235132486981, 58.238388045684786, 58.962266927929385, 60.59895275991553, 60.03130474640527, 60.95168802046201, 60.74909249689442, 60.51788117262136, 59.426066142476074, 59.672658115019566, 59.58841040533365, 59.535372449898475, 59.234471822418605, 57.88199099234541, 57.55517821095484, 57.091807923978905, 55.99899207380163, 55.47235774853064, 55.13597831570616, 54.88818553226556, 55.580822469295995, 54.604675639557804, 51.92798020726902, 52.13189282809286, 52.318955985013496, 52.16475740521177, 52.376708804372214, 52.53714479298515, 52.019375774017924, 51.41006552349257, 51.113397190371415, 51.60690747964943, 50.85633270407148, 49.91163883770975, 49.759486994000575, 50.560436057518764, 51.18703759989584, 51.0333087198768, 50.70656347003184, 49.53336205819175, 49.8731741460878, 48.25248331297989, 49.06050565123112, 50.96448951220852, 51.032468410847606, 50.9697200887467, 50.72950092340104, 50.84805023807254, 50.94491493464186, 50.796005999771644, 46.584620633176826, 44.053224751763615, 47.30340738558694, 49.59540076849495, 50.10949658525362, 51.27452052271473, 53.28732578153532, 54.711057897776726, 55.115785181078095, 48.37437745025324, 46.24291443588363, 49.02134644687096, 48.643740313389216, 51.227300535133494, 57.439363717458, 59.47589301209183, 61.365208802587695, 63.60551647230896, 64.51360739319563, 64.34963339692328, 64.6333333490907, 56.20096610302742, 26.521420011011248, 36.05909481540042, 56.910260087772045, 63.05565745574306, 63.87450113244978, 64.02890224998029, 64.12085537625957, 63.84323256569257, 62.411172873812475, 62.2441364497041, 63.86284402271001, 66.23990014602074, 68.12959688819835])

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