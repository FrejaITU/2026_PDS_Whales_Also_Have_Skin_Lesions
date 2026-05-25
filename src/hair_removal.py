import cv2
import numpy as np
import os
import pandas as pd
from pathlib import Path


def load_image(path):
    image = cv2.imread(path)

    if image is None:
        raise FileNotFoundError(f"Could not read image: {os.path.abspath(path)}")

    return image


def get_kernel_size(hair_coverage):
    if hair_coverage < 0.005:
        return None
    elif hair_coverage <= 0.035:
        return 15
    else:
        return 25


def create_hair_mask(image, kernel_size):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (kernel_size, kernel_size)
    )

    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

    combined = cv2.addWeighted(
        blackhat, 0.7,
        tophat, 0.3,
        0
    )

    hair_mask = cv2.adaptiveThreshold(
        combined,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        -2
    )

    closing_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    hair_mask = cv2.morphologyEx(
        hair_mask,
        cv2.MORPH_CLOSE,
        closing_kernel
    )

    return hair_mask


def remove_hair(image, hair_mask):
    return cv2.inpaint(
        image,
        hair_mask,
        inpaintRadius=3,
        flags=cv2.INPAINT_TELEA
    )


def process_folder(image_folder, csv_path, output_folder, output_mask_folder):
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(output_mask_folder, exist_ok=True)

    df = pd.read_csv(csv_path)

    for _, row in df.iterrows():
        filename = row["filename"]
        hair_coverage = row["hair_coverage"]

        image_path = os.path.join(image_folder, filename)
        output_path = os.path.join(output_folder, filename)

        name, ext = os.path.splitext(filename)
        mask_output_path = os.path.join(output_mask_folder, f"{name}_hair_mask.png")

        try:
            image = load_image(image_path)

            kernel_size = get_kernel_size(hair_coverage)

            if kernel_size is None:
                print(f"{filename}: skipping hair removal")

                cv2.imwrite(output_path, image)

                # save an empty mask for skipped images
                empty_mask = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.imwrite(mask_output_path, empty_mask)

                continue

            print(f"{filename}: coverage={hair_coverage:.5f}, kernel={kernel_size}")

            hair_mask = create_hair_mask(image, kernel_size)

            # save the generated hair mask
            cv2.imwrite(mask_output_path, hair_mask)

            cleaned = remove_hair(image, hair_mask)

            cv2.imwrite(output_path, cleaned)

        except FileNotFoundError as e:
            print("Skipping:", e)


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

image_folder = DATA_DIR / "imgs"
mask_folder = DATA_DIR / "masks"
csv_path = DATA_DIR / "hair_coverage.csv"
output_mask_folder = DATA_DIR / "masks_hair_removed"

output_mask_folder.mkdir(parents=True, exist_ok=True)

process_folder(
    image_folder=image_folder,
    mask_folder=mask_folder,
    output_folder=output_mask_folder,
    csv_path=csv_path
)