import cv2
import matplotlib.pyplot as plt
from pathlib import Path

base_dir = Path(__file__).resolve().parent

examples = [
    {
        "title": "Low hair",
        "original": "../../data/imgs/PAT_962_1823_942.png",
        "mask": "../../data/masks/PAT_962_1823_942_mask.png",
        "cleaned": "../../data/imgs_hair_removed/PAT_962_1823_942.png"
    },
    {
        "title": "Heavy hair",
        "original": "../../data/imgs/PAT_747_1409_116.png",
        "mask": "../../data/masks/PAT_747_1409_116_mask.png",
        "cleaned": "../../data/imgs_hair_removed/PAT_747_1409_116.png"
    }
]

fig, ax = plt.subplots(2, 3, figsize=(12, 12))

for row, ex in enumerate(examples):

    original = cv2.imread(ex["original"])
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(ex["mask"], cv2.IMREAD_GRAYSCALE)

    cleaned = cv2.imread(ex["cleaned"])
    cleaned = cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB)

    ax[row, 0].imshow(original)
    ax[row, 0].set_title(f'{ex["title"]} - Original')

    ax[row, 1].imshow(mask, cmap="gray")
    ax[row, 1].set_title("Lesion mask")

    ax[row, 2].imshow(cleaned)
    ax[row, 2].set_title("Hair removed")

    for col in range(3):
        ax[row, col].axis("off")
    # add a PAT label for this row (extract from filename)
    fname = Path(ex["original"]).stem
    # use the full PAT identifier (e.g. PAT_962_1823_942) but display as 'PAT 962_1823_942'
    if fname.startswith("PAT_"):
        pat_label = "PAT " + fname[len("PAT_"):]
    else:
        pat_label = fname
    bbox = ax[row, 0].get_position()
    x = bbox.x0 - 0.02
    y = bbox.y0 + bbox.height / 2
    fig.text(x, y, pat_label, va="center", ha="right", fontsize=12)
    

plt.tight_layout()
plt.subplots_adjust(left=0.12)

data_dir = base_dir.parent.parent / "data"
data_dir.mkdir(parents=True, exist_ok=True)
plots_dir = data_dir / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)
output_path = plots_dir / "figure_hair_removal.png"

plt.savefig(str(output_path), dpi=300, bbox_inches="tight")
print(f"Saved plot to {output_path}")

plt.show()