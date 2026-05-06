import cv2
import numpy as np
import json
import csv
from pathlib import Path


def load_image(image_path):
    """
    Load an image from the given file path.
    """

    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    return image


def segment_lesion(image):
    """
    Create an approximate mask of the skin lesion.

    This method assumes that the lesion is darker than the surrounding skin.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    _, threshold = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    cleaned = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

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
    """
    Calculate how much of the image is covered by the lesion.
    """

    total_pixels = lesion_mask.shape[0] * lesion_mask.shape[1]
    lesion_pixels = cv2.countNonZero(lesion_mask)

    coverage_percentage = (lesion_pixels / total_pixels) * 100

    return lesion_pixels, total_pixels, coverage_percentage


def clean_hair_with_tophat(
    image,
    kernel_size=17,
    threshold=12,
    radius=5,
    min_hair_pixels=50,
    min_hair_percentage=0.02
):
    """
    Detect and remove light/white hair using top-hat filtering.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    (
        tophat,
        mask,
        cleaned_image,
        light_hair_detected,
        hair_pixels,
        hair_percentage,
        filtering_strategy
    ) = removeHairTophat(
        img_org=image,
        img_gray=gray,
        kernel_size=kernel_size,
        threshold=threshold,
        radius=radius,
        min_hair_pixels=min_hair_pixels,
        min_hair_percentage=min_hair_percentage
    )

    return (
        tophat,
        mask,
        cleaned_image,
        light_hair_detected,
        hair_pixels,
        hair_percentage,
        filtering_strategy
    )


def removeHairTophat(
    img_org,
    img_gray,
    kernel_size=17,
    threshold=12,
    radius=5,
    min_hair_pixels=50,
    min_hair_percentage=0.02
):
    """
    Detect light hairs and remove them if enough hair is present.

    Top-hat morphology highlights bright thin structures on a darker background.
    """

    kernel = cv2.getStructuringElement(
        cv2.MORPH_CROSS,
        (kernel_size, kernel_size)
    )

    tophat = cv2.morphologyEx(
        img_gray,
        cv2.MORPH_TOPHAT,
        kernel
    )

    _, mask = cv2.threshold(
        tophat,
        threshold,
        255,
        cv2.THRESH_BINARY
    )

    hair_pixels = cv2.countNonZero(mask)
    total_pixels = mask.shape[0] * mask.shape[1]
    hair_percentage = (hair_pixels / total_pixels) * 100

    light_hair_detected = (
        hair_pixels >= min_hair_pixels and
        hair_percentage >= min_hair_percentage
    )

    if light_hair_detected:
        img_out = cv2.inpaint(
            img_org,
            mask,
            radius,
            cv2.INPAINT_TELEA
        )
        filtering_strategy = "tophat_cross_kernel"
    else:
        img_out = img_org.copy()
        filtering_strategy = "none_no_light_hair_detected"

    return (
        tophat,
        mask,
        img_out,
        light_hair_detected,
        hair_pixels,
        hair_percentage,
        filtering_strategy
    )


def analyze_skin_lesion(image_path, images_folder, results_folder=None):
    """
    Analyze one skin lesion image using top-hat hair removal.
    """

    image_path = Path(image_path)
    images_folder = Path(images_folder)
    results_folder = Path(results_folder)

    images_folder.mkdir(parents=True, exist_ok=True)
    results_folder.mkdir(parents=True, exist_ok=True)

    image = load_image(image_path)

    lesion_mask = segment_lesion(image)

    lesion_pixels, total_pixels, lesion_coverage = calculate_lesion_coverage(
        lesion_mask
    )

    (
        tophat_filtered,
        hair_mask,
        cleaned_image,
        light_hair_detected,
        hair_pixels,
        hair_percentage_of_image,
        filtering_strategy
    ) = clean_hair_with_tophat(
        image,
        kernel_size=17,
        threshold=12,
        radius=5,
        min_hair_pixels=50,
        min_hair_percentage=0.02
    )

    hair_pixels_total = cv2.countNonZero(hair_mask)
    total_pixels = image.shape[0] * image.shape[1]

    hair_percentage_of_image = (hair_pixels_total / total_pixels) * 100

    hair_mask_inside_lesion = cv2.bitwise_and(hair_mask, lesion_mask)
    hair_pixels_inside_lesion = cv2.countNonZero(hair_mask_inside_lesion)

    if lesion_pixels > 0:
        hair_percentage_of_lesion = (hair_pixels_inside_lesion / lesion_pixels) * 100
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

    cleaned_image_path = images_folder / f"{base_name}_cleaned.png"
    lesion_mask_path = images_folder / f"{base_name}_lesion_mask.png"
    hair_mask_path = images_folder / f"{base_name}_hair_mask.png"
    tophat_filtered_path = images_folder / f"{base_name}_tophat_filtered.png"

    cv2.imwrite(str(cleaned_image_path), cleaned_image)
    cv2.imwrite(str(lesion_mask_path), lesion_mask)
    cv2.imwrite(str(hair_mask_path), hair_mask)
    cv2.imwrite(str(tophat_filtered_path), tophat_filtered)

    info = {
        "input_image": str(image_path),

        "image_width": image.shape[1],
        "image_height": image.shape[0],
        "total_pixels": int(total_pixels),
        "lesion_pixels": int(lesion_pixels),
        "lesion_coverage_percentage": round(float(lesion_coverage), 3),

        "light_hair_detected": bool(light_hair_detected),

        "hair_pixels_total": int(hair_pixels_total),
        "hair_percentage_of_image": round(float(hair_percentage_of_image), 3),

        "hair_pixels_inside_lesion": int(hair_pixels_inside_lesion),
        "hair_percentage_of_lesion": round(float(hair_percentage_of_lesion), 3),

        "hair_class": hair_class,
        "filtering_strategy": filtering_strategy,

        "tophat_kernel_size": 17,
        "tophat_threshold": 12,
        "inpaint_radius": 5,
        "min_hair_pixels_required": 50,
        "min_hair_percentage_required": 0.02
    }

    print(f"Finished: {image_path.name}")

    return info


def save_combined_csv(all_results, output_path):
    """
    Save all image analysis results into one CSV file.
    """

    if len(all_results) == 0:
        return

    fieldnames = list(all_results[0].keys())

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()

        for result in all_results:
            writer.writerow(result)


if __name__ == "__main__":

    input_folder = Path("../results/blackhat_images")  # Process the cleaned images from blackhat step

    results_folder = Path("../results/tophat_results")
    images_folder = Path("../results/tophat_images")

    if not input_folder.exists():
        raise FileNotFoundError(f"The input folder does not exist: {input_folder}")

    if not input_folder.is_dir():
        raise NotADirectoryError(f"This is not a folder: {input_folder}")

    images_folder.mkdir(parents=True, exist_ok=True)
    results_folder.mkdir(parents=True, exist_ok=True)

    image_files = [
        file for file in input_folder.iterdir()
        if file.name.endswith("_cleaned.png")
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
                images_folder=images_folder,
                results_folder=results_folder
            )

            all_results.append(results)

        except Exception as e:
            print(f"Failed to process {image_file.name}: {e}")

    combined_json_path = results_folder / "all_tophat_analysis.json"
    combined_csv_path = results_folder / "all_tophat_analysis.csv"

    with open(combined_json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)

    save_combined_csv(all_results, combined_csv_path)

    print("\nFinished processing images.")
    print(f"Number of successful images: {len(all_results)}")
    print(f"Processed images folder: {images_folder}")
    print(f"Results folder: {results_folder}")
    print(f"Combined JSON saved to: {combined_json_path}")
    print(f"Combined CSV saved to: {combined_csv_path}")