# Metrics
import torch
import torchmetrics 
from torchmetrics.functional.classification import multilabel_confusion_matrix
import warnings
import rasterio as rio
from src.utils import get_finetune_config
from src.vit_spatial_spectral import ViTSpatialSpectral

# Load model
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
SEED = 5
dataset_name = "hypso"
config = get_finetune_config(
    f"configs/finetune_config_{dataset_name}.yaml",
    "configs/config.yaml",
    SEED,
    device,
)

## USER PARAMETERS ## 
config.checkpoint_path = "checkpoints/finetuned_ViTSpatialSpectral_10ep_enmap.pth" 
num_classes = config.n_classes
# Path to tiles. The folder specified should have tiles AND ground truth tiles, where the name of the tiles are tile0_hypso and tile0_hypso_gt, for example tile 0
test_path = "C:/Users/akuru/Documents/ECEE 7370 Advanced Comp Vision/sea_cloud_land/sample_test_folder/10-20220912_CaptureDL_00_doha_2022_09_12T06_38_29/"
test_tile = test_path + "tile0_hypso.TIF"
test_tile_gt = test_path + "tile0_hypso_gt.TIF"

model = ViTSpatialSpectral(
            image_size=config.image_size - config.patch_sub,
            spatial_patch_size=config.patch_size,
            spectral_patch_size=config.band_patch_size,
            num_classes=config.n_classes,
            dim=config.transformer_dim,
            depth=config.transformer_depth,
            heads=config.transformer_n_heads,
            mlp_dim=config.transformer_mlp_dim,
            dropout=config.transformer_dropout,
            emb_dropout=config.transformer_emb_dropout,
            channels=config.n_bands,
            spectral_pos=config.spectral_pos,
            spectral_pos_embed=config.spectral_pos_embed,
            blockwise_patch_embed=config.blockwise_patch_embed,
            spectral_only=config.spectral_only,
            pixelwise=config.pixelwise,
            pos_embed_len=config.pos_embed_len,
        )
model.load_state_dict(torch.load(config.checkpoint_path, map_location=torch.device('cpu'))["model_state_dict"])
model.to(device)
model.eval()


# Intersection over Union / jaccard index
jaccard = torchmetrics.JaccardIndex(
            task="multiclass", 
            threshold=0.5, 
            num_classes=num_classes,
            average="macro",
        )

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="Dataset has no geotransform, gcps, or rpcs. The identity matrix be returned.")
    with rio.open(test_tile_gt, num_threads=4) as f:
        label = f.read()[0]

prediction = model(test_tile_gt)
jaccard(prediction, label)

# Confusion matrix
confusion_matrix = multilabel_confusion_matrix(prediction, label, num_classes=num_classes)
