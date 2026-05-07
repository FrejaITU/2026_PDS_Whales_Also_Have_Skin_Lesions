import cv2
import numpy as np
import os
import pandas as pd
from glob import glob


def calculate_hair_coverage(image_path, mask_path):
    image = cv2.imread(image_path)
    lesion_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    if lesion_mask is None:
        raise FileNotFoundError(f"Could not read mask: {mask_path}")
    if lesion_mask.shape != image.shape[:2]:
        lesion_mask = cv2.resize(
            lesion_mask,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

    lesion_mask = (lesion_mask > 127).astype(np.uint8)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    sobel_x = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_mag = cv2.normalize(sobel_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    laplacian = cv2.Laplacian(enhanced, cv2.CV_64F)
    laplacian = np.absolute(laplacian)
    laplacian = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    combined = cv2.addWeighted(sobel_mag, 0.5, laplacian, 0.5, 0)

    _, hair_mask = cv2.threshold(combined, 60, 255, cv2.THRESH_BINARY)
    hair_mask = (hair_mask > 0).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_OPEN, kernel)

    hair_inside = np.logical_and(hair_mask > 0, lesion_mask > 0)

    hair_pixels = np.sum(hair_inside)
    lesion_pixels = np.sum(lesion_mask)

    if lesion_pixels == 0:
        return 0

    return hair_pixels / lesion_pixels


def process_folder(image_folder, mask_folder, output_csv):
    results = []

    image_paths = glob(os.path.join(image_folder, "*.png"))

    for image_path in image_paths:
        filename = os.path.basename(image_path)

        if "_mask" in filename:
            continue

        name, ext = os.path.splitext(filename)
        mask_filename = f"{name}_mask{ext}"
        mask_path = os.path.join(mask_folder, mask_filename)

        try:
            coverage = calculate_hair_coverage(image_path, mask_path)

            results.append({
                "filename": filename,
                "mask_filename": mask_filename,
                "hair_coverage": coverage
            })

            print(f"{filename}: {coverage:.5f}")

        except FileNotFoundError as e:
            print("Skipping:", e)

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)

    print("Saved CSV to:", output_csv)


if __name__ == "__main__":
    image_folder = "../data/imgs"
    mask_folder = "../data/masks"
    output_csv = "../data/hair_coverage.csv"

    process_folder(image_folder, mask_folder, output_csv)