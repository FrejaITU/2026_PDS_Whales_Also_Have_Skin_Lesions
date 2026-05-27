# Projects in Data Science (2026)

#### Setup

You must have a metadata csv file including patient IDs (column name ("")) with corresponding image IDs (column name "img_id") and a corresponding diagnosis (column name "")

Insert all lesion images in /data/imgs
Insert all masks in /data/masks
Insert the metadata csv in /data as "metadata.csv"


#### Feature extraction

Run /src/Hair_removal/hair_coverage.py
Run /src/Hair_removal/hair_removal.py

Run /src/split_mask_components.py

Run /src/Features/run_all_features for each mask type (normal, biggest, top3) and both hair and no hair combination - 6 total.

This will output 6 CSVs with feature scores for each permutation.




