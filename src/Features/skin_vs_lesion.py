from pathlib import Path
import numpy as np
import pandas as pd
from skimage import color, io


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"

IMG_DIR = DATA_DIR / "imgs"
ORIGINAL_MASK_DIR = DATA_DIR / "masks"
COMPONENT_MASK_DIR = DATA_DIR / "masks_top3_split_components"

CSV_PATH = DATA_DIR / "metadata_top3_split_components_features.csv"


FEATURE_COLUMNS = [
    "lesion_red_share",
    "lesion_green_share",
    "lesion_blue_share",
    "lesion_skin_red_diff",
    "lesion_skin_green_diff",
    "lesion_skin_blue_diff",
    "lesion_skin_rgb_distance",
]


def empty_result():
    return {column: np.nan for column in FEATURE_COLUMNS}


def load_image(path):
    image = io.imread(path)

    if image.ndim == 3 and image.shape[-1] == 4:
        image = image[:, :, :3]

    return image.astype(float)


def load_mask(path):
    mask = io.imread(path)

    if mask.ndim == 3:
        mask = color.rgb2gray(mask)

    return mask > 0


def rgb_features(image, lesion_mask, original_mask):
    skin_mask = ~original_mask

    if not np.any(lesion_mask):
        return empty_result()

    if not np.any(skin_mask):
        return empty_result()

    lesion_rgb_mean = np.mean(image[lesion_mask], axis=0)
    skin_rgb_mean = np.mean(image[skin_mask], axis=0)

    lesion_rgb_sum = np.sum(lesion_rgb_mean)

    if lesion_rgb_sum == 0:
        return empty_result()

    rgb_diff = lesion_rgb_mean - skin_rgb_mean

    return {
        "lesion_red_share": lesion_rgb_mean[0] / lesion_rgb_sum,
        "lesion_green_share": lesion_rgb_mean[1] / lesion_rgb_sum,
        "lesion_blue_share": lesion_rgb_mean[2] / lesion_rgb_sum,
        "lesion_skin_red_diff": rgb_diff[0],
        "lesion_skin_green_diff": rgb_diff[1],
        "lesion_skin_blue_diff": rgb_diff[2],
        "lesion_skin_rgb_distance": np.linalg.norm(rgb_diff),
    }


metadata = pd.read_csv(CSV_PATH)

for column in FEATURE_COLUMNS:
    if column not in metadata.columns:
        metadata[column] = np.nan


for i, row in metadata.iterrows():
    if (i + 1) % 25 == 0:
        print(f"Processed {i + 1} rows")

    image_path = IMG_DIR / row["img_id"]
    original_mask_path = ORIGINAL_MASK_DIR / row["original_mask"]
    component_mask_path = COMPONENT_MASK_DIR / row["component_mask"]

    if not image_path.exists():
        print("Missing image:", image_path)
        continue

    if not original_mask_path.exists():
        print("Missing original mask:", original_mask_path)
        continue

    if not component_mask_path.exists():
        print("Missing component mask:", component_mask_path)
        continue

    image = load_image(image_path)
    original_mask = load_mask(original_mask_path)
    component_mask = load_mask(component_mask_path)

    result = rgb_features(
        image=image,
        lesion_mask=component_mask,
        original_mask=original_mask,
    )

    for column, value in result.items():
        metadata.at[i, column] = value


metadata.to_csv(CSV_PATH, index=False)
print("Finished RGB lesion-skin features")