import cv2
import numpy as np
import json
import csv
import pandas as pd
from pathlib import Path


def load_image(image_path):
    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    return image


def segment_lesion(image):
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
    total_pixels = lesion_mask.shape[0] * lesion_mask.shape[1]
    lesion_pixels = cv2.countNonZero(lesion_mask)

    coverage_percentage = (lesion_pixels / total_pixels) * 100

    return lesion_pixels, total_pixels, coverage_percentage


def create_blackhat_mask(image, kernel_size=17, threshold=17):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_CROSS,
        (kernel_size, kernel_size)
    )

    blackhat = cv2.morphologyEx(
        gray,
        cv2.MORPH_BLACKHAT,
        kernel
    )

    _, mask = cv2.threshold(
        blackhat,
        threshold,
        255,
        cv2.THRESH_BINARY
    )

    return blackhat, mask


def load_manual_hair_annotations(annotation_csv_path):
    """
    Loads your hand-annotation CSV and calculates the average human hair score.

    Expected columns:
        img_id
        hair_1, hair_2, hair_3, hair_4, hair_5
    """

    df = pd.read_csv(annotation_csv_path)

    hair_columns = [
        col for col in df.columns
        if col.startswith("hair_")
    ]

    for col in hair_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["manual_hair_average"] = df[hair_columns].mean(axis=1, skipna=True)
    df["manual_hair_annotation_count"] = df[hair_columns].notna().sum(axis=1)

    def manual_class(avg):
        if pd.isna(avg):
            return None
        elif avg < 0.5:
            return "none"
        elif avg < 1.5:
            return "small"
        elif avg < 2.5:
            return "medium"
        else:
            return "large"

    df["manual_hair_class"] = df["manual_hair_average"].apply(manual_class)

    # Change this if you want a stricter human definition.
    # 0.5 means: if the average human score is at least small, hair removal should be considered.
    df["manual_should_clean"] = df["manual_hair_average"] >= 0.5

    return df


def get_manual_annotation_for_image(image_path, manual_annotations_df):
    """
    Finds the manual annotation row for one image.
    Handles normal names and names ending in _cleaned.
    """

    if manual_annotations_df is None:
        return {}

    image_name = image_path.name
    image_name_without_cleaned = image_name.replace("_cleaned", "")

    matched_rows = manual_annotations_df[
        (manual_annotations_df["img_id"] == image_name) |
        (manual_annotations_df["img_id"] == image_name_without_cleaned)
    ]

    if len(matched_rows) == 0:
        return {
            "manual_hair_1": None,
            "manual_hair_2": None,
            "manual_hair_3": None,
            "manual_hair_4": None,
            "manual_hair_5": None,
            "manual_hair_average": None,
            "manual_hair_annotation_count": None,
            "manual_hair_class": None,
            "manual_should_clean": None
        }

    row = matched_rows.iloc[0]

    return {
        "manual_hair_1": row.get("hair_1", None),
        "manual_hair_2": row.get("hair_2", None),
        "manual_hair_3": row.get("hair_3", None),
        "manual_hair_4": row.get("hair_4", None),
        "manual_hair_5": row.get("hair_5", None),
        "manual_hair_average": row.get("manual_hair_average", None),
        "manual_hair_annotation_count": row.get("manual_hair_annotation_count", None),
        "manual_hair_class": row.get("manual_hair_class", None),
        "manual_should_clean": row.get("manual_should_clean", None)
    }


def otsu_threshold_from_values(values, bins=256):
    """
    Otsu threshold for 1D values, for example hair_percentage_of_image.

    Returns a threshold in the same unit as the input.
    If the input is percent, the output is percent.
    """

    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]

    if len(values) == 0:
        raise ValueError("No valid values for Otsu threshold.")

    if np.min(values) == np.max(values):
        return float(values[0])

    counts, bin_edges = np.histogram(values, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    total = counts.sum()
    sum_total = np.sum(counts * bin_centers)

    sum_background = 0.0
    weight_background = 0

    best_variance = -1
    best_threshold = bin_centers[0]

    for i in range(len(counts)):
        weight_background += counts[i]

        if weight_background == 0:
            continue

        weight_foreground = total - weight_background

        if weight_foreground == 0:
            break

        sum_background += counts[i] * bin_centers[i]

        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground

        between_class_variance = (
            weight_background *
            weight_foreground *
            (mean_background - mean_foreground) ** 2
        )

        if between_class_variance > best_variance:
            best_variance = between_class_variance
            best_threshold = bin_centers[i]

    return float(best_threshold)


def calculate_hair_measurements(image, lesion_mask, hair_mask):
    total_pixels = image.shape[0] * image.shape[1]

    hair_pixels_total = cv2.countNonZero(hair_mask)
    hair_percentage_of_image = (hair_pixels_total / total_pixels) * 100

    lesion_pixels = cv2.countNonZero(lesion_mask)

    hair_mask_inside_lesion = cv2.bitwise_and(hair_mask, lesion_mask)
    hair_pixels_inside_lesion = cv2.countNonZero(hair_mask_inside_lesion)

    if lesion_pixels > 0:
        hair_percentage_of_lesion = (hair_pixels_inside_lesion / lesion_pixels) * 100
    else:
        hair_percentage_of_lesion = 0

    return {
        "hair_pixels_total": int(hair_pixels_total),
        "hair_percentage_of_image": float(hair_percentage_of_image),
        "hair_pixels_inside_lesion": int(hair_pixels_inside_lesion),
        "hair_percentage_of_lesion": float(hair_percentage_of_lesion)
    }


def classify_hair_amount(hair_percentage_of_lesion):
    if hair_percentage_of_lesion < 0.5:
        return "none"
    elif hair_percentage_of_lesion < 2.0:
        return "small"
    elif hair_percentage_of_lesion < 5.0:
        return "medium"
    else:
        return "large"


def analyze_skin_lesion(
    image_path,
    images_folder,
    otsu_blackhat_threshold_percent,
    manual_annotations_df=None,
    kernel_size=17,
    blackhat_threshold=17,
    inpaint_radius=7
):
    image_path = Path(image_path)
    images_folder = Path(images_folder)
    images_folder.mkdir(parents=True, exist_ok=True)

    image = load_image(image_path)

    lesion_mask = segment_lesion(image)

    lesion_pixels, total_pixels, lesion_coverage = calculate_lesion_coverage(
        lesion_mask
    )

    blackhat_filtered, hair_mask = create_blackhat_mask(
        image,
        kernel_size=kernel_size,
        threshold=blackhat_threshold
    )

    hair_measurements = calculate_hair_measurements(
        image=image,
        lesion_mask=lesion_mask,
        hair_mask=hair_mask
    )

    hair_percentage_of_image = hair_measurements["hair_percentage_of_image"]
    hair_percentage_of_lesion = hair_measurements["hair_percentage_of_lesion"]

    hair_class = classify_hair_amount(hair_percentage_of_lesion)

    blackhat_applied_by_otsu = hair_percentage_of_image >= otsu_blackhat_threshold_percent

    if blackhat_applied_by_otsu:
        cleaned_image = cv2.inpaint(
            image,
            hair_mask,
            inpaint_radius,
            cv2.INPAINT_TELEA
        )
        filtering_strategy = "blackhat_applied_otsu"
    else:
        cleaned_image = image.copy()
        filtering_strategy = "none_below_otsu_threshold"

    base_name = image_path.stem.replace("_cleaned", "")

    cleaned_image_path = images_folder / f"{base_name}_cleaned.png"
    lesion_mask_path = images_folder / f"{base_name}_lesion_mask.png"
    hair_mask_path = images_folder / f"{base_name}_hair_mask.png"
    blackhat_filtered_path = images_folder / f"{base_name}_blackhat_filtered.png"

    cv2.imwrite(str(cleaned_image_path), cleaned_image)
    cv2.imwrite(str(lesion_mask_path), lesion_mask)
    cv2.imwrite(str(hair_mask_path), hair_mask)
    cv2.imwrite(str(blackhat_filtered_path), blackhat_filtered)

    manual_info = get_manual_annotation_for_image(
        image_path=image_path,
        manual_annotations_df=manual_annotations_df
    )

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

        "hair_pixels_total": int(hair_measurements["hair_pixels_total"]),
        "hair_percentage_of_image": round(float(hair_percentage_of_image), 3),
        "hair_pixels_inside_lesion": int(hair_measurements["hair_pixels_inside_lesion"]),
        "hair_percentage_of_lesion": round(float(hair_percentage_of_lesion), 3),

        "manual_hair_1": manual_info.get("manual_hair_1"),
        "manual_hair_2": manual_info.get("manual_hair_2"),
        "manual_hair_3": manual_info.get("manual_hair_3"),
        "manual_hair_4": manual_info.get("manual_hair_4"),
        "manual_hair_5": manual_info.get("manual_hair_5"),
        "manual_hair_average": manual_info.get("manual_hair_average"),
        "manual_hair_annotation_count": manual_info.get("manual_hair_annotation_count"),
        "manual_hair_class": manual_info.get("manual_hair_class"),
        "manual_should_clean": manual_info.get("manual_should_clean"),

        "hair_class": hair_class,
        "blackhat_applied_by_otsu": bool(blackhat_applied_by_otsu),
        "otsu_blackhat_threshold_percent": round(float(otsu_blackhat_threshold_percent), 3),
        "filtering_strategy": filtering_strategy,

        "blackhat_kernel_size": kernel_size,
        "blackhat_threshold": blackhat_threshold,
        "inpaint_radius": inpaint_radius
    }

    print(f"Finished: {image_path.name}")
    print(f"Hair percentage of image: {hair_percentage_of_image:.3f}%")
    print(f"Otsu threshold: {otsu_blackhat_threshold_percent:.3f}%")
    print(f"Filtering used: {filtering_strategy}")

    return info


def preview_blackhat_measurement(
    image_path,
    manual_annotations_df=None,
    kernel_size=17,
    blackhat_threshold=17
):
    """
    First pass:
    Calculate hair coverage only.
    No cleaning is performed here.
    """

    image_path = Path(image_path)
    image = load_image(image_path)

    lesion_mask = segment_lesion(image)
    lesion_pixels, total_pixels, lesion_coverage = calculate_lesion_coverage(lesion_mask)

    blackhat_filtered, hair_mask = create_blackhat_mask(
        image,
        kernel_size=kernel_size,
        threshold=blackhat_threshold
    )

    hair_measurements = calculate_hair_measurements(
        image=image,
        lesion_mask=lesion_mask,
        hair_mask=hair_mask
    )

    manual_info = get_manual_annotation_for_image(
        image_path=image_path,
        manual_annotations_df=manual_annotations_df
    )

    row = {
        "input_image": str(image_path),
        "image_name": image_path.name,

        "image_width": image.shape[1],
        "image_height": image.shape[0],
        "total_pixels": int(total_pixels),
        "lesion_pixels": int(lesion_pixels),
        "lesion_coverage_percentage": round(float(lesion_coverage), 3),

        "hair_pixels_total": int(hair_measurements["hair_pixels_total"]),
        "hair_percentage_of_image": round(float(hair_measurements["hair_percentage_of_image"]), 3),
        "hair_pixels_inside_lesion": int(hair_measurements["hair_pixels_inside_lesion"]),
        "hair_percentage_of_lesion": round(float(hair_measurements["hair_percentage_of_lesion"]), 3),

        "manual_hair_1": manual_info.get("manual_hair_1"),
        "manual_hair_2": manual_info.get("manual_hair_2"),
        "manual_hair_3": manual_info.get("manual_hair_3"),
        "manual_hair_4": manual_info.get("manual_hair_4"),
        "manual_hair_5": manual_info.get("manual_hair_5"),
        "manual_hair_average": manual_info.get("manual_hair_average"),
        "manual_hair_annotation_count": manual_info.get("manual_hair_annotation_count"),
        "manual_hair_class": manual_info.get("manual_hair_class"),
        "manual_should_clean": manual_info.get("manual_should_clean"),

        "blackhat_kernel_size": kernel_size,
        "blackhat_threshold": blackhat_threshold
    }

    return row


def validate_otsu_against_manual(preview_results, otsu_threshold_percent):
    """
    Compares the Otsu decision with the manual annotation decision.
    """

    df = pd.DataFrame(preview_results)

    if "manual_should_clean" not in df.columns:
        return {}

    df = df.dropna(subset=["manual_should_clean"])

    if len(df) == 0:
        return {}

    manual_should_clean = df["manual_should_clean"].astype(bool)
    otsu_should_clean = df["hair_percentage_of_image"] >= otsu_threshold_percent

    true_positive = int(((otsu_should_clean == True) & (manual_should_clean == True)).sum())
    true_negative = int(((otsu_should_clean == False) & (manual_should_clean == False)).sum())
    false_positive = int(((otsu_should_clean == True) & (manual_should_clean == False)).sum())
    false_negative = int(((otsu_should_clean == False) & (manual_should_clean == True)).sum())

    accuracy = (true_positive + true_negative) / len(df)

    precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive) > 0
        else 0
    )

    recall = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative) > 0
        else 0
    )

    return {
        "otsu_threshold_percent": round(float(otsu_threshold_percent), 3),
        "manual_rule": "manual_hair_average >= 0.5",
        "manual_rows_used": int(len(df)),
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "accuracy": round(float(accuracy), 3),
        "precision": round(float(precision), 3),
        "recall": round(float(recall), 3)
    }


def save_combined_csv(all_results, output_path):
    if len(all_results) == 0:
        return

    fieldnames = list(all_results[0].keys())

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in all_results:
            writer.writerow(result)

def make_json_serializable(obj):
    """
    Converts NumPy/pandas values into normal Python values
    so json.dump() can save them.
    """

    if isinstance(obj, dict):
        return {
            key: make_json_serializable(value)
            for key, value in obj.items()
        }

    if isinstance(obj, list):
        return [
            make_json_serializable(value)
            for value in obj
        ]

    if isinstance(obj, tuple):
        return tuple(
            make_json_serializable(value)
            for value in obj
        )

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.bool_):
        return bool(obj)

    if pd.isna(obj):
        return None

    return obj

if __name__ == "__main__":
    input_folder = Path("../data/imgs")
    annotation_csv_path = Path("../data/Old/annotations_combined.csv")

    results_folder = Path("../results/blackhat_results")
    images_folder = Path("../results/blackhat_images")

    kernel_size = 17
    blackhat_threshold = 17
    inpaint_radius = 7

    if not input_folder.exists():
        raise FileNotFoundError(f"The input folder does not exist: {input_folder}")

    if not input_folder.is_dir():
        raise NotADirectoryError(f"This is not a folder: {input_folder}")

    if not annotation_csv_path.exists():
        raise FileNotFoundError(f"The annotation CSV does not exist: {annotation_csv_path}")

    images_folder.mkdir(parents=True, exist_ok=True)
    results_folder.mkdir(parents=True, exist_ok=True)

    image_files = [
        file for file in input_folder.iterdir()
        if file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
    ]

    if len(image_files) == 0:
        print("No image files found in the input folder.")
        exit()

    manual_annotations_df = load_manual_hair_annotations(annotation_csv_path)

    # First pass: calculate blackhat hair coverage for every image.
    preview_results = []

    print("\nCalculating blackhat hair coverage for Otsu threshold...")

    for image_file in image_files:
        try:
            preview_row = preview_blackhat_measurement(
                image_path=image_file,
                manual_annotations_df=manual_annotations_df,
                kernel_size=kernel_size,
                blackhat_threshold=blackhat_threshold
            )

            preview_results.append(preview_row)

        except Exception as e:
            print(f"Failed preview for {image_file.name}: {e}")

    preview_csv_path = results_folder / "blackhat_preview_before_otsu.csv"
    save_combined_csv(preview_results, preview_csv_path)

    hair_percentages = [
        row["hair_percentage_of_image"]
        for row in preview_results
    ]

    otsu_blackhat_threshold_percent = otsu_threshold_from_values(
        hair_percentages,
        bins=256
    )

    validation = validate_otsu_against_manual(
        preview_results=preview_results,
        otsu_threshold_percent=otsu_blackhat_threshold_percent
    )

    validation_json_path = results_folder / "blackhat_otsu_validation.json"

    with open(validation_json_path, "w", encoding="utf-8") as f:
        json.dump(make_json_serializable(validation), f, indent=4)

    print("\nOtsu threshold found")
    print("--------------------")
    print(f"Otsu blackhat threshold: {otsu_blackhat_threshold_percent:.3f}%")

    if validation:
        print("\nValidation against manual annotations")
        print("-------------------------------------")
        print(f"Manual rule: {validation['manual_rule']}")
        print(f"Accuracy: {validation['accuracy']}")
        print(f"Precision: {validation['precision']}")
        print(f"Recall: {validation['recall']}")

    # Second pass: actually clean images using the Otsu threshold.
    all_results = []

    print("\nCleaning images using Otsu threshold...")

    for image_file in image_files:
        print(f"\nProcessing: {image_file.name}")

        try:
            results = analyze_skin_lesion(
                image_path=image_file,
                images_folder=images_folder,
                otsu_blackhat_threshold_percent=otsu_blackhat_threshold_percent,
                manual_annotations_df=manual_annotations_df,
                kernel_size=kernel_size,
                blackhat_threshold=blackhat_threshold,
                inpaint_radius=inpaint_radius
            )

            all_results.append(results)

        except Exception as e:
            print(f"Failed to process {image_file.name}: {e}")

    combined_json_path = results_folder / "all_blackhat_analysis.json"
    combined_csv_path = results_folder / "all_blackhat_analysis.csv"

    with open(combined_json_path, "w", encoding="utf-8") as f:
        json.dump(make_json_serializable(all_results), f, indent=4)

    save_combined_csv(all_results, combined_csv_path)

    print("\nFinished processing images.")
    print(f"Number of successful images: {len(all_results)}")
    print(f"Processed images folder: {images_folder}")
    print(f"Results folder: {results_folder}")
    print(f"Preview CSV saved to: {preview_csv_path}")
    print(f"Otsu validation saved to: {validation_json_path}")
    print(f"Combined JSON saved to: {combined_json_path}")
    print(f"Combined CSV saved to: {combined_csv_path}")