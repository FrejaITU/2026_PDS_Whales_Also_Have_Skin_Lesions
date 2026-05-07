import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

base_dir = Path(__file__).resolve().parent
coverage_csv = base_dir.parent / "data" / "hair_coverage.csv"
annotations_csv = base_dir.parent / "data" / "Old" / "annotations_combined.csv"

coverage_df = pd.read_csv(coverage_csv)
annotations_df = pd.read_csv(annotations_csv)

# Use the same key in both files for a correct merge
annotations_df = annotations_df.rename(columns={"img_id": "filename"})

hair_cols = [f"hair_{i}" for i in range(1, 6)]
annotations_df[hair_cols] = annotations_df[hair_cols].apply(pd.to_numeric, errors="coerce")

annotations_df["mean_hair_label"] = annotations_df[hair_cols].mean(axis=1)

modes = annotations_df[hair_cols].mode(axis=1, numeric_only=True)
annotations_df["majority_hair_label"] = modes[0].fillna(0).astype(int)

merged = coverage_df.merge(
    annotations_df[["filename", "mean_hair_label", "majority_hair_label"]],
    on="filename",
    how="inner",
)

print("Merged rows:", len(merged))
print(merged[["filename", "hair_coverage", "mean_hair_label", "majority_hair_label"]].head(10))
print(merged[["hair_coverage", "mean_hair_label", "majority_hair_label"]].describe())
print(merged.groupby("majority_hair_label")["hair_coverage"].mean())
print(merged.groupby("majority_hair_label")["hair_coverage"].median())

merged = merged[merged["majority_hair_label"].isin([0, 1, 2, 3])].copy()
merged = merged[merged["hair_coverage"] > 0].copy()

merged["majority_hair_label"] = merged["majority_hair_label"].astype(int)

palette = {
    0: "blue",
    1: "orange",
    2: "green",
    3: "red",
}

plt.figure(figsize=(8, 5))

sns.violinplot(
    data=merged,
    x="majority_hair_label",
    y="hair_coverage",
    hue="majority_hair_label",
    palette=palette,
    cut=0,
    legend=False
)

sns.boxplot(
    data=merged,
    x="majority_hair_label",
    y="hair_coverage",
    width=0.15,
    showfliers=False,
    boxprops={"facecolor": "white", "edgecolor": "black"},
    whiskerprops={"color": "black"},
    capprops={"color": "black"},
    medianprops={"color": "black"}
)

plt.ylim(0, 0.05)
plt.xlabel("Annotation label")
plt.ylabel("Hair coverage")
plt.title("Hair coverage by annotation label")

plots_dir = base_dir.parent / "data" / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)
output_path = plots_dir / "hair_coverage_violin_boxplot.png"

plt.savefig(str(output_path), dpi=300, bbox_inches="tight")
print(f"Saved plot to {output_path}")
plt.show()