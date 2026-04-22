#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
from PIL import Image
from pathlib import Path

metadata = pd.read_csv("../data/metadata.csv")

for i, patient in enumerate(metadata.itertuples(), start=1):
    if i % 25 == 0:
        print(f"Processed {i} imgs.")
    if not Path(f"../data/masks/{patient.img_id[0:-4]}_mask.png").exists():
        continue
    img = Image.open(f"../data/masks/{patient.img_id[0:-4]}_mask.png")
    binary_img = img.copy().convert("1")
    binary_img.save(f"../data/binary_masks/{patient.img_id[0:-4]}_mask.png")


# In[ ]:




