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
data_path = BASE_DIR / "../data"
img_dir = data_path / "imgs"
no_hair_dir = data_path / "imgs_hair_removed"
mask_dir = data_path / "masks"
top3_dir = data_path / "masks_top3_split_components"
biggest_dir = data_path / "masks_biggest_component"


metadata = pd.read_csv(data_path / "metadata.csv")

hc.process_folder()
hr.process_folder()
smc.run()


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





def run(img_path, mask_path, csv_name):
    for i, patient in enumerate(metadata.itertuples(), start=1):
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
                if column_name not in metadata.columns:
                    metadata[column_name] = pd.NA
                metadata.at[idx, column_name] = value
    metadata.to_csv(data_path / csv_name, index=False)

run(img_dir, mask_dir, "features")
run(img_dir, biggest_dir, "biggest_features")
run(img_dir, top3_dir, "top3_features")
run(no_hair_dir, mask_dir, "no_hair_features")
run(no_hair_dir, biggest_dir, "biggest_no_hair_features")
run(no_hair_dir, top3_dir, "top3_no_hair_features")