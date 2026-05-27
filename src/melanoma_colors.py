### melanoma colors and count
import numpy as np
from skimage import color
from skimage.segmentation import slic
from skimage.morphology import binary_erosion, disk


REFERENCE_HSV = {
    "white": np.array([0.72, 0.13, 0.85]),
    "black": np.array([0.02, 0.27, 0.16]),
    "red": np.array([0.01, 0.86, 0.46]),
    "light_brown": np.array([0.07, 0.90, 0.64]),
    "dark_brown": np.array([0.05, 0.96, 0.53]),
    "blue_gray": np.array([0.69, 0.22, 0.55]),
}


def slic_segmentation(
    image,
    mask,
    compactness=0.1,
    sigma=1,
    enforce_connectivity=True
):
    n_segments = max(15, min(100, round(mask.sum() / 400))) # There is a superpixel per at less 400 pixel.

    segments = slic(
        image,
        n_segments=n_segments,
        compactness=compactness,
        sigma=sigma,
        mask=mask,
        start_label=1,
        channel_axis=-1,
        convert2lab=True,
        enforce_connectivity=enforce_connectivity
    )

    return segments


def hsv_distance(segment_colors_hsv, reference_color_hsv):
    hue_diff = np.abs(segment_colors_hsv[:, 0] - reference_color_hsv[0])
    hue_diff = np.minimum(hue_diff, 1 - hue_diff)

    sat_diff = segment_colors_hsv[:, 1] - reference_color_hsv[1]
    val_diff = segment_colors_hsv[:, 2] - reference_color_hsv[2]

    distance = np.sqrt(hue_diff**2 + sat_diff**2 + val_diff**2)

    return distance


def count_melanoma_colors(image, mask, segments, tolerance=0.15, erosion_radius=3):
    image_hsv = color.rgb2hsv(image)

    inner_mask = binary_erosion(mask, disk(erosion_radius))

    if not np.any(inner_mask):
        inner_mask = mask

    segment_colors_hsv = []

    for segment_id in np.unique(segments[mask]):
        current_segment_mask = (segments == segment_id) & inner_mask # i only want an average color over my inner mask. 

        if not np.any(current_segment_mask):
            continue

        mean_hsv = image_hsv[current_segment_mask].mean(axis=0)
        segment_colors_hsv.append(mean_hsv)

    if len(segment_colors_hsv) == 0:
        return 0, ""

    segment_colors_hsv = np.array(segment_colors_hsv)

    found_colors = []

    for color_name, reference_color in REFERENCE_HSV.items():
        distances = hsv_distance(segment_colors_hsv, reference_color)

        if np.any(distances <= tolerance):
            found_colors.append(color_name)

    color_count = len(found_colors)
    found_colors_string = ", ".join(found_colors)

    return color_count, found_colors_string


def melanoma_colors_score(image, mask):
    segments = slic_segmentation(image, mask)
    color_count, found_colors = count_melanoma_colors(image, mask, segments)

    return {
        "melanoma_color_count": color_count,
        "melanoma_colors": found_colors
    }