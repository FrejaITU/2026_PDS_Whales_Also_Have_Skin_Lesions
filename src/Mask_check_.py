### Check original dataset

from pathlib import Path
import numpy as np
import pandas as pd
from skimage import io
from skimage.morphology import remove_small_objects

# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
data_path = BASE_DIR / "data"
img_dir = data_path / "imgs"
mask_dir = data_path / "masks"

metadata = pd.read_csv(data_path / "metadata_(for_scores).csv")

# ----------------------------------------
# Check masks
# ----------------------------------------

missing_masks = []
wrong_size_masks = []
empty_masks = []
empty_masks_after_cleaning = []

for patient in metadata.itertuples():
    img_id = patient.img_id

    img_path = img_dir / img_id
    mask_path = mask_dir / f"{Path(img_id).stem}_mask.png"

    if not mask_path.exists():
        missing_masks.append(mask_path.name)
        continue

    image = io.imread(img_path)
    mask = io.imread(mask_path)

    if image.shape[:2] != mask.shape[:2]:
        wrong_size_masks.append({
            "image": img_id,
            "mask": mask_path.name,
            "image_shape": image.shape[:2],
            "mask_shape": mask.shape[:2],
        })
        continue


    mask = mask > 0

    if not np.any(mask):
        empty_masks.append(mask_path.name)
        continue

    cleaned_mask = remove_small_objects(mask, min_size=50)

    if not np.any(cleaned_mask):
        empty_masks_after_cleaning.append(mask_path.name)

mask_info = f"""Missing masks:
{missing_masks}

Masks with wrong size:
{wrong_size_masks}

Empty masks:
{empty_masks}

Empty masks after removing small objects:
{empty_masks_after_cleaning}
"""

print(mask_info)

(data_path / "mask_check_output.tex").write_text(mask_info)