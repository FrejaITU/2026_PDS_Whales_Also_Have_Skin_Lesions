import numpy as np
from skimage import color, io


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

def rgb_features(image, component_mask, original_mask):
    skin_mask = ~original_mask

    if not np.any(component_mask):
        return empty_result()

    if not np.any(skin_mask):
        return empty_result()

    lesion_rgb_mean = np.mean(image[component_mask], axis=0)
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

