# Projects in Data Science (2026)

#### Setup

You must have a metadata csv file including patient IDs (column name "patient_id") with corresponding image IDs (column name "img_id") and a corresponding diagnosis (column name "diagnostic")

Insert all lesion images in data/imgs/

Insert all masks in data/masks/

Insert the metadata csv in data/ as "metadata.csv"


#### Feature extraction and evaluation

Run src/extract_features.py

This will output 6 CSVs with feature scores in data/features/

In main.py, set load_models to True or False

Run main.py

This will output the predictions in results/predictions/ if load_models is True, and output the PKL files and top 10 features for each model if False.
