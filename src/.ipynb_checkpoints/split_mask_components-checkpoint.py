from pathlib import Path

import numpy as np
import pandas as pd
from skimage import color, io
from skimage.measure import label
from skimage.morphology import binary_closing


# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

MASK_DIR = DATA_DIR / "masks"
BIGGEST_DIR = DATA_DIR / "masks_biggest_component"
TOP3_DIR = DATA_DIR / "masks_top3_split_components"

BIGGEST_DIR.mkdir(exist_ok=True)
TOP3_DIR.mkdir(exist_ok=True)

METADATA_PATH = DATA_DIR / "metadata_reduced.csv"
BIGGEST_CSV_PATH = DATA_DIR / "metadata_biggest_component.csv"
TOP3_CSV_PATH = DATA_DIR / "metadata_top3_split_components.csv"


# --------------------------------------------------
# Settings
# --------------------------------------------------
CONNECTIVITY = 2
MIN_EXTRA_RATIO = 0.10
CLOSING_FOOTPRINT = np.ones((3, 3), dtype=bool)

CSV_COLUMNS = [
    "img_id",
    "patient_id",
    "lesion_id",
    "gender",
    "skin_cancer_diagnosis",
    "diagnostic",
    "biopsed",
    "original_mask",
    "component_mask",
]


# --------------------------------------------------
# Functions
# --------------------------------------------------
def load_mask(mask_path):
    mask = io.imread(mask_path)

    if mask.ndim == 3:
        if mask.shape[-1] == 4:
            mask = mask[:, :, :3]

        mask = color.rgb2gray(mask)

    return mask > 0.5


def get_components(mask):
    """
    Returns components sorted from biggest to smallest.

    Closing is used only to decide grouping.
    The saved component masks still use the original mask pixels.
    """

    closed_mask = binary_closing(mask, footprint=CLOSING_FOOTPRINT)

    labeled_mask, num_components = label(
        closed_mask,
        connectivity=CONNECTIVITY,
        return_num=True
    )

    components = []

    for component_id in range(1, num_components + 1):
        grouped_component = labeled_mask == component_id
        component_mask = grouped_component & mask
        components.append(component_mask)

    components.sort(key=lambda component: component.sum(), reverse=True)

    return components


def select_top_components(components):
    biggest = components[0]
    min_area = biggest.sum() * MIN_EXTRA_RATIO

    selected = [biggest]

    for component in components[1:3]:
        if component.sum() >= min_area:
            selected.append(component)

    return selected


def save_mask(mask, path):
    io.imsave(
        path,
        mask.astype(np.uint8) * 255,
        check_contrast=False
    )


def make_csv_row(metadata_row, original_mask_name, component_mask_name):
    return {
        "img_id": metadata_row.img_id,
        "patient_id": metadata_row.patient_id,
        "lesion_id": metadata_row.lesion_id,
        "gender": metadata_row.gender,
        "skin_cancer_diagnosis": metadata_row.skin_cancer_diagnosis,
        "diagnostic": metadata_row.diagnostic,
        "biopsed": metadata_row.biopsed,
        "original_mask": original_mask_name,
        "component_mask": component_mask_name,
    }


# --------------------------------------------------
# Main
# --------------------------------------------------
metadata = pd.read_csv(METADATA_PATH)

biggest_rows = []
top3_rows = []

for i, metadata_row in enumerate(metadata.itertuples(index=False), start=1):
    if i % 25 == 0:
        print(f"Processed {i} rows.")

    img_id = metadata_row.img_id
    img_stem = Path(img_id).stem

    original_mask_name = f"{img_stem}_mask.png"
    original_mask_path = MASK_DIR / original_mask_name

    mask = load_mask(original_mask_path)
    components = get_components(mask)

    # --------------------------------------------------
    # 1. Biggest component only
    # --------------------------------------------------
    biggest_mask = components[0]

    biggest_output_name = original_mask_name
    biggest_output_path = BIGGEST_DIR / biggest_output_name

    save_mask(biggest_mask, biggest_output_path)

    biggest_rows.append(
        make_csv_row(
            metadata_row,
            original_mask_name,
            biggest_output_name
        )
    )

    # --------------------------------------------------
    # 2. Top 3 split components
    # --------------------------------------------------
    top_components = select_top_components(components)

    for component_index, component_mask in enumerate(top_components, start=1):
        component_output_name = f"{img_stem}_component_{component_index}.png"
        component_output_path = TOP3_DIR / component_output_name

        save_mask(component_mask, component_output_path)

        top3_rows.append(
            make_csv_row(
                metadata_row,
                original_mask_name,
                component_output_name
            )
        )


# --------------------------------------------------
# Save CSVs
# --------------------------------------------------
pd.DataFrame(biggest_rows, columns=CSV_COLUMNS).to_csv(
    BIGGEST_CSV_PATH,
    index=False
)

pd.DataFrame(top3_rows, columns=CSV_COLUMNS).to_csv(
    TOP3_CSV_PATH,
    index=False
)

print("Finished.")