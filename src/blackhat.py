import cv2
import numpy as np
import json
import csv
from pathlib import Path


def load_image(image_path):
    """
    Load an image from the given file path.

    OpenCV returns None if the image cannot be loaded,
    so we check for that and raise an error.
    """

    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    return image


def segment_lesion(image):
    """
    Create an approximate mask of the skin lesion.

    This method assumes that the lesion is darker than the surrounding skin.
    It converts the image to grayscale, thresholds the dark region,
    cleans the mask, and keeps the largest detected region as the lesion.
    """

    # Convert image from BGR color to grayscale.
    # Thresholding is easier on one intensity channel.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blur the image to reduce small noise before thresholding.
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Use Otsu thresholding to automatically separate dark and light areas.
    # THRESH_BINARY_INV makes darker areas white in the mask.
    _, threshold = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Create an elliptical kernel for morphological cleaning.
    # Elliptical kernels fit lesion-like shapes better than square kernels.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    # Opening removes small isolated noise.
    cleaned = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)

    # Closing fills small gaps or holes inside the lesion area.
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    # Find all detected white regions in the cleaned binary mask.
    contours, _ = cv2.findContours(
        cleaned,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Create an empty mask with the same height and width as the grayscale image.
    lesion_mask = np.zeros(gray.shape, dtype=np.uint8)

    # If no contours are found, return an empty lesion mask.
    if len(contours) == 0:
        return lesion_mask

    # Assume the largest detected region is the lesion.
    largest_contour = max(contours, key=cv2.contourArea)

    # Fill the largest contour with white pixels.
    cv2.drawContours(lesion_mask, [largest_contour], -1, 255, thickness=-1)

    return lesion_mask


def calculate_lesion_coverage(lesion_mask):
    """
    Calculate how much of the image is covered by the lesion.

    Returns:
    - lesion_pixels: number of white pixels in the lesion mask
    - total_pixels: total number of pixels in the image
    - coverage_percentage: percentage of the image covered by the lesion
    """

    total_pixels = lesion_mask.shape[0] * lesion_mask.shape[1]
    lesion_pixels = cv2.countNonZero(lesion_mask)

    coverage_percentage = (lesion_pixels / total_pixels) * 100

    return lesion_pixels, total_pixels, coverage_percentage


def clean_hair_with_blackhat(
    image,
    kernel_size=17,
    threshold=17,
    radius=7,
    min_hair_pixels=50,
    min_hair_percentage=0.02
):
    """
    Detect and remove dark hair using black-hat filtering.

    This wrapper function converts the image to grayscale and then calls
    removeHair(), which performs the actual black-hat detection and inpainting.
    """

    # Convert to grayscale because black-hat morphology works on intensity.
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

    return (
        blackhat,
        mask,
        cleaned_image,
        black_hair_detected,
        hair_pixels,
        hair_percentage,
        filtering_strategy
    )


def removeHair(
    img_org,
    img_gray,
    kernel_size=17,
    threshold=17,
    radius=7,
    min_hair_pixels=50,
    min_hair_percentage=0.02
):
    """
    Detect dark hairs and remove them if enough hair is present.

    Black-hat morphology highlights dark thin structures on a lighter background.
    This is useful for detecting dark hairs on skin.

    If enough hair pixels are detected, the code uses inpainting to replace
    those pixels using nearby surrounding pixels.
    """

    # Create a cross-shaped kernel for black-hat morphology.
    # The kernel size controls the scale of structures that are detected.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_CROSS,
        (kernel_size, kernel_size)
    )

    # Apply black-hat filtering.
    # This highlights dark structures such as hair.
    blackhat = cv2.morphologyEx(
        img_gray,
        cv2.MORPH_BLACKHAT,
        kernel
    )

    # Threshold the black-hat image to create a binary hair mask.
    # Pixels above the threshold are treated as hair.
    _, mask = cv2.threshold(
        blackhat,
        threshold,
        255,
        cv2.THRESH_BINARY
    )

    # Count how many pixels were detected as hair.
    hair_pixels = cv2.countNonZero(mask)
    total_pixels = mask.shape[0] * mask.shape[1]
    hair_percentage = (hair_pixels / total_pixels) * 100

    # Only remove hair if enough hair pixels are detected.
    # This avoids unnecessary inpainting when little or no hair exists.
    black_hair_detected = (
        hair_pixels >= min_hair_pixels and
        hair_percentage >= min_hair_percentage
    )

    if black_hair_detected:
        # Inpaint the original image where the hair mask is white.
        # TELEA fills masked pixels using nearby image information.
        img_out = cv2.inpaint(
            img_org,
            mask,
            radius,
            cv2.INPAINT_TELEA
        )
        filtering_strategy = "blackhat_cross_kernel"
    else:
        # If no meaningful hair is detected, keep the original image unchanged.
        img_out = img_org.copy()
        filtering_strategy = "none_no_black_hair_detected"

    return blackhat, mask, img_out, black_hair_detected, hair_pixels, hair_percentage, filtering_strategy


def analyze_skin_lesion(image_path, images_folder, results_folder=None):
    """
    Analyze one skin lesion image.

    The function:
    1. Loads the image.
    2. Segments the lesion.
    3. Calculates lesion size.
    4. Detects and removes dark hair.
    5. Measures how much hair overlaps with the lesion.
    6. Classifies hair amount as none, small, medium, or large.
    7. Saves output images.
    8. Returns all measurements in a dictionary.
    """

    image_path = Path(image_path)
    images_folder = Path(images_folder)
    results_folder = Path(results_folder)

    # Make sure output folders exist.
    images_folder.mkdir(parents=True, exist_ok=True)
    results_folder.mkdir(parents=True, exist_ok=True)

    # Load original image.
    image = load_image(image_path)

    # Create lesion mask.
    lesion_mask = segment_lesion(image)

    # Calculate lesion coverage in the full image.
    lesion_pixels, total_pixels, lesion_coverage = calculate_lesion_coverage(
        lesion_mask
    )

    # Detect hair and create a cleaned version of the image.
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

    # Recalculate hair amount over the full image.
    hair_pixels_total = cv2.countNonZero(hair_mask)
    total_pixels = image.shape[0] * image.shape[1]

    hair_percentage_of_image = (hair_pixels_total / total_pixels) * 100

    # Find hair pixels that are inside the lesion area.
    hair_mask_inside_lesion = cv2.bitwise_and(hair_mask, lesion_mask)
    hair_pixels_inside_lesion = cv2.countNonZero(hair_mask_inside_lesion)

    # Calculate hair percentage relative to the lesion area.
    if lesion_pixels > 0:
        hair_percentage_of_lesion = (hair_pixels_inside_lesion / lesion_pixels) * 100
    else:
        hair_percentage_of_lesion = 0

    # Classify hair amount based on how much of the lesion is covered by hair.
    if hair_percentage_of_lesion < 0.5:
        hair_class = "none"
    elif hair_percentage_of_lesion < 2.0:
        hair_class = "small"
    elif hair_percentage_of_lesion < 5.0:
        hair_class = "medium"
    else:
        hair_class = "large"

    # Use the original filename without extension for output names.
    base_name = image_path.stem

    # Define output paths for generated images.
    cleaned_image_path = images_folder / f"{base_name}_cleaned.png"
    lesion_mask_path = images_folder / f"{base_name}_lesion_mask.png"
    hair_mask_path = images_folder / f"{base_name}_hair_mask.png"
    blackhat_filtered_path = images_folder / f"{base_name}_blackhat_filtered.png"

    # Save generated images.
    cv2.imwrite(str(cleaned_image_path), cleaned_image)
    cv2.imwrite(str(lesion_mask_path), lesion_mask)
    cv2.imwrite(str(hair_mask_path), hair_mask)
    cv2.imwrite(str(blackhat_filtered_path), blackhat_filtered)

    # Store file paths, measurements, classifications, and parameters.
    info = {
        "input_image": str(image_path),

        "image_width": image.shape[1],
        "image_height": image.shape[0],
        "total_pixels": int(total_pixels),
        "lesion_pixels": int(lesion_pixels),
        "lesion_coverage_percentage": round(float(lesion_coverage), 3),

        "black_hair_detected": bool(black_hair_detected),

        "hair_pixels_total": int(hair_pixels_total),
        "hair_percentage_of_image": round(float(hair_percentage_of_image), 3),

        "hair_pixels_inside_lesion": int(hair_pixels_inside_lesion),
        "hair_percentage_of_lesion": round(float(hair_percentage_of_lesion), 3),

        "hair_class": hair_class,
        "filtering_strategy": filtering_strategy,

        "blackhat_kernel_size": 17,
        "blackhat_threshold": 17,
        "inpaint_radius": 7,
        "min_hair_pixels_required": 50,
        "min_hair_percentage_required": 0.02
    }

    print(f"Finished: {image_path.name}")

    return info


def save_combined_csv(all_results, output_path):
    """
    Save all image analysis results into one CSV file.

    Each row is one processed image.
    Each column is one stored measurement, file path, or parameter.
    """

    if len(all_results) == 0:
        return

    # Use the dictionary keys as CSV column names.
    fieldnames = list(all_results[0].keys())

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()

        for result in all_results:
            writer.writerow(result)


if __name__ == "__main__":
    """
    Main script.

    This part runs when the file is executed directly.
    It processes every image in the input folder and saves:
    - cleaned images
    - lesion masks
    - hair masks
    - black-hat filtered images
    - combined JSON results
    - combined CSV results
    """

    # Folder containing the original images.
    input_folder = Path("../data/imgs")

    # Folder where JSON and CSV results will be saved.
    results_folder = Path("../data/blackhat_results")

    # Folder where generated images will be saved.
    images_folder = Path("../data/blackhat_images")

    # Check that the input folder exists.
    if not input_folder.exists():
        raise FileNotFoundError(f"The input folder does not exist: {input_folder}")

    # Check that the input path is actually a folder.
    if not input_folder.is_dir():
        raise NotADirectoryError(f"This is not a folder: {input_folder}")

    # Create output folders if they do not exist.
    images_folder.mkdir(parents=True, exist_ok=True)
    results_folder.mkdir(parents=True, exist_ok=True)

    # Collect all supported image files from the input folder.
    image_files = [
        file for file in input_folder.iterdir()
        if file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
    ]

    # Stop the script if no image files are found.
    if len(image_files) == 0:
        print("No image files found in the input folder.")
        exit()

    # This list stores the result dictionary from each successfully processed image.
    all_results = []

    # Process each image one at a time.
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
            # If one image fails, print the error and continue with the next image.
            print(f"Failed to process {image_file.name}: {e}")

    # Define combined result output files.
    combined_json_path = results_folder / "all_blackhat_analysis.json"
    combined_csv_path = results_folder / "all_blackhat_analysis.csv"

    # Save all results together as JSON.
    with open(combined_json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)

    # Save all results together as CSV.
    save_combined_csv(all_results, combined_csv_path)

    print("\nFinished processing images.")
    print(f"Number of successful images: {len(all_results)}")
    print(f"Processed images folder: {images_folder}")
    print(f"Results folder: {results_folder}")
    print(f"Combined JSON saved to: {combined_json_path}")
    print(f"Combined CSV saved to: {combined_csv_path}")