# Projects in Data Science (2026)

#### Setup

You must have a metadata csv file including patient IDs (column name ("")) with corresponding image IDs (column name "img_id") and a corresponding diagnosis (column name "")

Insert all lesion images in /data/imgs

Insert all masks in /data/masks

Insert the metadata csv in /data as "metadata.csv"


#### Feature extraction

Run /src/extract_features

This will output 6 CSVs with feature scores in /data/features
