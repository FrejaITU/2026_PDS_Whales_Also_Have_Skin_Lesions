import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

base_dir = Path(__file__).resolve().parent

# ---------------------------------------------------
# LOAD IMAGE + MASK
# ---------------------------------------------------

image_path = "../../data/imgs/PAT_747_1409_116.png"
mask_path = "../../data/masks/PAT_747_1409_116_mask.png"

image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

lesion_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

lesion_mask = cv2.resize(
    lesion_mask,
    (image.shape[1], image.shape[0]),
    interpolation=cv2.INTER_NEAREST
)

lesion_mask = (lesion_mask > 127).astype(np.uint8)

# ---------------------------------------------------
# CLAHE
# ---------------------------------------------------

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

enhanced = clahe.apply(gray)

# ---------------------------------------------------
# SOBEL + LAPLACIAN
# ---------------------------------------------------

sobel_x = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)

sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
sobel_mag = cv2.normalize(
    sobel_mag,
    None,
    0,
    255,
    cv2.NORM_MINMAX
).astype(np.uint8)

laplacian = cv2.Laplacian(enhanced, cv2.CV_64F)
laplacian = np.absolute(laplacian)

laplacian = cv2.normalize(
    laplacian,
    None,
    0,
    255,
    cv2.NORM_MINMAX
).astype(np.uint8)

combined = cv2.addWeighted(
    sobel_mag,
    0.5,
    laplacian,
    0.5,
    0
)

_, hair_mask = cv2.threshold(
    combined,
    80,
    255,
    cv2.THRESH_BINARY
)

hair_mask = (hair_mask > 0).astype(np.uint8)

hair_inside = hair_mask * lesion_mask

# ---------------------------------------------------
# PLOT
# ---------------------------------------------------

fig, ax = plt.subplots(1, 5, figsize=(18, 4))
fig.suptitle("PAT_747_1409_116", fontsize=16, y=1.02)

ax[0].imshow(image)
ax[0].set_title("Original")

ax[1].imshow(gray, cmap="gray")
ax[1].set_title("Gray")

ax[2].imshow(enhanced, cmap="gray")
ax[2].set_title("CLAHE")

ax[3].imshow(hair_mask, cmap="gray")
ax[3].set_title("Hair mask")

ax[4].imshow(hair_inside, cmap="gray")
ax[4].set_title("Hair inside lesion")

for a in ax:
    a.axis("off")

plt.tight_layout()
plt.subplots_adjust(top=0.85)

data_dir = base_dir.parent.parent / "results" / "figures"
data_dir.mkdir(parents=True, exist_ok=True)
output_path = data_dir / "figure_x_pipeline.png"

plt.savefig(str(output_path), dpi=300, bbox_inches="tight")
print(f"Saved plot to {output_path}")


plt.show()