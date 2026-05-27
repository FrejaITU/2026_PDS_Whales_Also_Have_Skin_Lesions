from pathlib import Path
import numpy as np
import pandas as pd
from skimage import color, io
from skimage.morphology import remove_small_objects

# --------------------------------------------------
# Change only this block when you want a new feature
# --------------------------------------------------
### import your script like you would a libary
import melanoma_colors as mcf
import get_asymmetry as ga
import get_compactness as gc
import hsv_variance as hsvv
import Mabrouk_asymmetry_score as mba
import convexity_score as cs
import new_mabrouk as nmba
import new_get_asymmetry as nga

### Add it to a dictonary
FEATURES = {
    "melanoma_colors": mcf.melanoma_colors_score,
    "get_asymmetry": ga.get_asymmetry,
    "get_compactness": gc.get_compactness,
    "hsv_variance": hsvv.hsv_var_score,
    "mabrouk_asymmetry": mba.Mabrouk_asymmetry,
    "convexity": cs.convexity_score,
    "new_mabrouk": nmba.Mabrouk_asymmetry,
    "new_get_asymmetry": nga.get_asymmetry
    }


# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
data_path = BASE_DIR / "../../data"
img_dir = data_path / "imgs"
mask_dir = data_path / "masks"
metadata_dir = data_path / "metadata.csv"
metadata = pd.read_csv(metadata_dir)

split_masks = False
if input("Are masks split? (y/n) ").lower in ("y", "yes", "1", "true"):
    split_masks = True

def load_image_and_mask(img_path, mask_path):
    image = io.imread(img_path)
    mask = io.imread(mask_path)

    if mask.ndim == 3:
        mask = color.rgb2gray(mask)

    mask = mask > 0
    mask = remove_small_objects(mask, min_size=50)

    if image.ndim == 3 and image.shape[-1] == 4:
        image = image[:, :, :3]

    return image, mask



for i, patient in enumerate(metadata.itertuples(), start=1):
    if i % 25 == 0:
        print(f"Processed {i} rows.")

    idx = patient.Index
    img_id = patient.img_id

    current_img_path = img_dir / img_id
    
    if split_mask:
        current_mask_path = mask_dir / patient.component_mask
    else:
        current_mask_path = mask_dir / f"{Path(img_id).stem}_mask.png"

    if not current_img_path.exists():
        print("Missing image:", img_id)
        continue

    if not current_mask_path.exists():
        print("Missing mask:", current_mask_path.name)
        continue

    image, mask = load_image_and_mask(current_img_path, current_mask_path)

    if mask.shape != image.shape[:2]:
        print("Skipping (shape mismatch):", current_img_path.name, mask.shape, image.shape)
        continue

    if not np.any(mask):
        print("Skipping (empty mask):", current_img_path.name)
        continue

    for feature, function in FEATURES.items():
        result = function(image, mask)
    
        for column_name, value in result.items():
            if column_name not in metadata.columns:
                metadata[column_name] = pd.NA
            metadata.at[idx, column_name] = value
        
    
metadata.to_csv(metadata_dir, index=False)
print("Finished running all features")