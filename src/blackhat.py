import cv2
import numpy as np
import json
import csv
from pathlib import Path


def load_image(image_path):
    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    return image


def segment_lesion(image):
    """
    Approximate lesion segmentation using grayscale thresholding.
    This works best when the lesion is darker than surrounding skin.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Smooth image to reduce noise
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Otsu thresholding
    _, threshold = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    cleaned = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    # Keep largest connected component as lesion
    contours, _ = cv2.findContours(
        cleaned,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    lesion_mask = np.zeros(gray.shape, dtype=np.uint8)

    if len(contours) == 0:
        return lesion_mask

    largest_contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(lesion_mask, [largest_contour], -1, 255, thickness=-1)

    return lesion_mask


def calculate_lesion_coverage(lesion_mask):
    total_pixels = lesion_mask.shape[0] * lesion_mask.shape[1]
    lesion_pixels = cv2.countNonZero(lesion_mask)

    coverage_percentage = (lesion_pixels / total_pixels) * 100

    return lesion_pixels, total_pixels, coverage_percentage

def keep_hair_like_components(mask):
    """
    Keep only thin, hair-like connected components.

    This prevents the lesion itself or large dark areas from being detected as hair.
    """

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    cleaned_mask = np.zeros_like(mask)

    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]

        if area == 0:
            continue

        aspect_ratio = max(w, h) / max(1, min(w, h))

        # Hair is usually thin and elongated.
        # These values can be adjusted depending on your images.
        is_long_enough = max(w, h) >= 8
        is_thin_enough = min(w, h) <= 12
        is_not_too_large = area <= 1200
        is_line_like = aspect_ratio >= 2.0

        if is_long_enough and is_thin_enough and is_not_too_large and is_line_like:
            cleaned_mask[labels == label] = 255

    return cleaned_mask

def make_line_kernel(length, angle):
    """
    Create a thin line-shaped kernel rotated by angle degrees.
    This is better for hair than a square kernel.
    """
    kernel = np.zeros((length, length), dtype=np.uint8)

    center = length // 2
    cv2.line(kernel, (center, 0), (center, length - 1), 1, 1)

    rotation_matrix = cv2.getRotationMatrix2D((center, center), angle, 1.0)
    rotated = cv2.warpAffine(kernel, rotation_matrix, (length, length))

    rotated = (rotated > 0).astype(np.uint8)

    return rotated

def clean_hair_with_blackhat(image, kernel_size=17, threshold=17, radius=7, min_hair_pixels=50, min_hair_percentage=0.02):
    """
    Uses black-hat only if dark hair is detected.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    (
        blackhat,
        mask,
        cleaned_image,
        black_hair_detected,
        hair_pixels,
        hair_percentage,
        filtering_strategy
    ) = removeHair(
        img_org=image,
        img_gray=gray,
        kernel_size=kernel_size,
        threshold=threshold,
        radius=radius,
        min_hair_pixels=min_hair_pixels,
        min_hair_percentage=min_hair_percentage
    )


    return (blackhat, mask, cleaned_image, black_hair_detected, hair_pixels, hair_percentage, filtering_strategy)


def classify_hair_amount(hair_mask, lesion_mask=None):
    hair_pixels = cv2.countNonZero(hair_mask)
    total_pixels = hair_mask.shape[0] * hair_mask.shape[1]

    hair_percentage = (hair_pixels / total_pixels) * 100

    if hair_percentage < 0.2:
        hair_class = "none"
    elif hair_percentage < 1.0:
        hair_class = "small"
    elif hair_percentage < 3.0:
        hair_class = "medium"
    else:
        hair_class = "large"

    return hair_pixels, hair_percentage, hair_class


def choose_filtering_strategy(blackhat_mask, tophat_mask, hair_class):
    blackhat_pixels = cv2.countNonZero(blackhat_mask)
    tophat_pixels = cv2.countNonZero(tophat_mask)

    if hair_class == "none":
        return "none"

    # For heavy hair, use both filters.
    if hair_class == "large":
        return "blackhat_and_tophat"

    if blackhat_pixels == 0 and tophat_pixels == 0:
        return "none"

    if blackhat_pixels > tophat_pixels * 1.3:
        return "blackhat"

    if tophat_pixels > blackhat_pixels * 1.3:
        return "tophat"

    return "blackhat_and_tophat"


def removeHair(img_org, img_gray, kernel_size=17, threshold=17, radius=7, min_hair_pixels=50, min_hair_percentage=0.02):
    """
    Detect dark hair using black-hat morphology.

    Only performs inpainting if enough black/dark hair is detected.

    min_hair_pixels:
        Minimum number of detected hair pixels required.

    min_hair_percentage:
        Minimum percentage of the image detected as hair.
        Example: 0.02 means 0.02% of the full image.
    """

    kernel = cv2.getStructuringElement(
        cv2.MORPH_CROSS,
        (kernel_size, kernel_size)
    )

    blackhat = cv2.morphologyEx(
        img_gray,
        cv2.MORPH_BLACKHAT,
        kernel
    )

    _, mask = cv2.threshold(
        blackhat,
        threshold,
        255,
        cv2.THRESH_BINARY
    )

    hair_pixels = cv2.countNonZero(mask)
    total_pixels = mask.shape[0] * mask.shape[1]
    hair_percentage = (hair_pixels / total_pixels) * 100

    black_hair_detected = (
        hair_pixels >= min_hair_pixels and
        hair_percentage >= min_hair_percentage
    )

    if black_hair_detected:
        img_out = cv2.inpaint(
            img_org,
            mask,
            radius,
            cv2.INPAINT_TELEA
        )
        filtering_strategy = "blackhat_cross_kernel"
    else:
        img_out = img_org.copy()
        filtering_strategy = "none_no_black_hair_detected"

    return blackhat, mask, img_out, black_hair_detected, hair_pixels, hair_percentage, filtering_strategy


def save_results_json(info, output_path):
    with open(output_path, "w") as f:
        json.dump(info, f, indent=4)


def save_results_csv(info, output_path):
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(["Parameter", "Value"])

        for key, value in info.items():
            writer.writerow([key, value])


def analyze_skin_lesion(image_path, output_folder="output"):
    image_path = Path(image_path)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    image = load_image(image_path)

    lesion_mask = segment_lesion(image)

    lesion_pixels, total_pixels, lesion_coverage = calculate_lesion_coverage(
        lesion_mask
    )

    (
        blackhat_filtered, 
        hair_mask, 
        cleaned_image, 
        black_hair_detected, 
        hair_pixels, 
        hair_percentage_of_image, 
        filtering_strategy
    ) = clean_hair_with_blackhat(
        image,
        kernel_size=17,
        threshold=17,
        radius=7,
        min_hair_pixels=50,
        min_hair_percentage=0.02
    )

    if lesion_pixels > 0:
        hair_percentage_of_lesion = (hair_pixels / lesion_pixels) * 100
    else:
        hair_percentage_of_lesion = 0

    if hair_percentage_of_lesion < 0.5:
        hair_class = "none"
    elif hair_percentage_of_lesion < 2.0:
        hair_class = "small"
    elif hair_percentage_of_lesion < 5.0:
        hair_class = "medium"
    else:
        hair_class = "large"

    base_name = image_path.stem

    cleaned_image_path = output_folder / f"{base_name}_cleaned.png"
    lesion_mask_path = output_folder / f"{base_name}_lesion_mask.png"
    hair_mask_path = output_folder / f"{base_name}_hair_mask.png"
    blackhat_filtered_path = output_folder / f"{base_name}_blackhat_filtered.png"
    json_path = output_folder / f"{base_name}_analysis.json"

    cv2.imwrite(str(cleaned_image_path), cleaned_image)
    cv2.imwrite(str(lesion_mask_path), lesion_mask)
    cv2.imwrite(str(hair_mask_path), hair_mask)
    cv2.imwrite(str(blackhat_filtered_path), blackhat_filtered)

    info = {
        "input_image": str(image_path),
        "cleaned_image": str(cleaned_image_path),
        "lesion_mask": str(lesion_mask_path),
        "hair_mask": str(hair_mask_path),
        "blackhat_filtered_image": str(blackhat_filtered_path),

        "image_width": image.shape[1],
        "image_height": image.shape[0],
        "total_pixels": int(total_pixels),
        "lesion_pixels": int(lesion_pixels),
        "lesion_coverage_percentage": round(float(lesion_coverage), 3),

        # Hair detection / cleaning results
        "black_hair_detected": bool(black_hair_detected),
        "hair_pixels": int(hair_pixels),
        "hair_percentage_of_image": round(float(hair_percentage_of_image), 3),
        "hair_percentage_of_lesion": round(float(hair_percentage_of_lesion), 3),
        "hair_class": hair_class,
        "filtering_strategy": filtering_strategy,

        # Parameters used
        "blackhat_kernel_size": 17,
        "blackhat_threshold": 17,
        "inpaint_radius": 7,
        "min_hair_pixels_required": 50,
        "min_hair_percentage_required": 0.02    
    }

    save_results_json(info, json_path)

    print(f"Finished: {image_path.name}")
    print(f"Hair class: {hair_class}")
    print("Filtering used: blackhat_cross_kernel")

    return info

def save_combined_csv(all_results, output_path):
    if len(all_results) == 0:
        return

    fieldnames = list(all_results[0].keys())

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()

        for result in all_results:
            writer.writerow(result)


if __name__ == "__main__":
    # Hardcode your input and output folders here
    input_folder = Path("../data/imgs") 
    output_folder = Path("../results/blackhat_output")

    if not input_folder.exists():
        raise FileNotFoundError(f"The input folder does not exist: {input_folder}")

    if not input_folder.is_dir():
        raise NotADirectoryError(f"This is not a folder: {input_folder}")

    output_folder.mkdir(parents=True, exist_ok=True)

    image_files = [
        file for file in input_folder.iterdir()
        if file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
    ]

    if len(image_files) == 0:
        print("No image files found in the input folder.")
        exit()

    all_results = []

    for image_file in image_files:
        print(f"\nProcessing: {image_file.name}")

        try:
            results = analyze_skin_lesion(
                image_path=image_file,
                output_folder=output_folder
            )

            all_results.append(results)

        except Exception as e:
            print(f"Failed to process {image_file.name}: {e}")

    combined_json_path = output_folder / "all_images_analysis.json"
    combined_csv_path = output_folder / "all_images_analysis.csv"

    with open(combined_json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)

    save_combined_csv(all_results, combined_csv_path)

    print("\nFinished processing images.")
    print(f"Number of successful images: {len(all_results)}")
    print(f"Output folder: {output_folder}")
    print(f"Combined JSON saved to: {combined_json_path}")
    print(f"Combined CSV saved to: {combined_csv_path}")