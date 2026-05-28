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
import get_compactness as gc
import hsv_variance as hsvv
import convexity_score as cs
import mabrouk as mba
import get_asymmetry as ga
import skin_vs_lesion as svl

import hair_removal as hr
import hair_coverage as hc
import split_mask_components as smc

### Add it to a dictonary
FEATURES = {
    "melanoma_colors": mcf.melanoma_colors_score,
    "get_asymmetry": ga.get_asymmetry,
    "get_compactness": gc.get_compactness,
    "hsv_variance": hsvv.hsv_var_score,
    "mabrouk_asymmetry": mba.Mabrouk_asymmetry,
    "convexity": cs.convexity_score,
    }


# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
data_path = BASE_DIR / "data"
features_dir = data_path / "features"
img_dir = data_path / "imgs"
no_hair_dir = data_path / "imgs_hair_removed"
mask_dir = data_path / "masks"
top3_dir = data_path / "masks_top3_split_components"
biggest_dir = data_path / "masks_biggest_component"

hc.process_folder()
hr.process_folder()
smc.run()

metadata = pd.read_csv(data_path / "metadata.csv")
metadata_biggest = pd.read_csv(data_path / "metadata_biggest_component.csv")
metadata_top3 = pd.read_csv(data_path / "metadata_top3_split_components.csv")

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





def run(img_path, mask_path, data, csv_name):
    for i, patient in enumerate(data.itertuples(), start=1):
        if i % 25 == 0:
            print(f"Processed {i} rows.")

        idx = patient.Index
        img_id = patient.img_id

        current_img_path = img_path / img_id
        if mask_path != mask_dir:
            current_mask_path = mask_path / patient.component_mask
        else:
            current_mask_path = mask_path / f"{Path(img_id).stem}_mask.png"

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
                if column_name not in data.columns:
                    data[column_name] = pd.NA
                data.at[idx, column_name] = value
        if mask_path == mask_dir:
            result = svl.rgb_features(image, mask, mask)
        else:
            original_mask = io.imread(mask_dir / f"{Path(img_id).stem}_mask.png")
            if original_mask.ndim == 3:
                original_mask = color.rgb2gray(original_mask)
            original_mask = original_mask > 0
            original_mask = remove_small_objects(original_mask, min_size=50)
            result = svl.rgb_features(image, mask, original_mask)
        for column_name, value in result.items():
                if column_name not in data.columns:
                    data[column_name] = pd.NA
                data.at[idx, column_name] = value
    data.to_csv(features_dir / csv_name, index=False)

run(img_dir, mask_dir, metadata, "features.csv")
run(img_dir, biggest_dir, metadata_biggest, "tbiggest_features.csv")
run(img_dir, top3_dir, metadata_top3, "top3_features.csv")
run(no_hair_dir, mask_dir, metadata, "no_hair_features.csv")
run(no_hair_dir, biggest_dir, metadata_biggest, "biggest_no_hair_features.csv")
run(no_hair_dir, top3_dir, metadata_top3, "top3_no_hair_features.csv")

print("Finished all feature extractions.")