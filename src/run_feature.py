from pathlib import Path
import numpy as np
import pandas as pd
from skimage import color, io

# --------------------------------------------------
# Change only this block when you want a new feature
# --------------------------------------------------
### import your script like you would a libary
import mask_components as mc
### Add it to a dictonary
FEATURES = {
    "mask_components": lambda image, mask: mc.mask_components(image, mask),
}
# Change the Feature name to choose yours
FEATURE_NAME = "mask_components"
FEATURE_FUNCTION = FEATURES[FEATURE_NAME]

# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
data_path = BASE_DIR / "data"
img_dir = data_path / "imgs"
mask_dir = data_path / "masks"

# Our CSV
metadata = pd.read_csv(data_path / "metadata_(for_scores).csv")

# Create the output column if it does not exist
if FEATURE_NAME not in metadata.columns:
    metadata[FEATURE_NAME] = np.nan


def load_image_and_mask(img_path, mask_path):
    image = io.imread(img_path)
    mask = io.imread(mask_path)

    if mask.ndim == 3:
        mask = color.rgb2gray(mask)

    mask = mask > 0

    if image.ndim == 3 and image.shape[-1] == 4:
        image = image[:, :, :3]

    return image, mask


for patient in metadata.itertuples():
    idx = patient.Index
    img_id = patient.img_id

    current_img_path = img_dir / img_id
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

    score = FEATURE_FUNCTION(image, mask)
    metadata.at[idx, FEATURE_NAME] = score

metadata.to_csv(data_path / "metadata_(for_scores).csv", index=False)
print(f"Finished running feature: {FEATURE_NAME}")