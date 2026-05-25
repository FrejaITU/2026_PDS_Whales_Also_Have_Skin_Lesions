#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

df = pd.read_csv("../data/metadata_(for_scores).csv")

df = df.drop(["img_id", "gender", "diagnostic", "biopsed", "melanoma_colors"], axis = 1)
df["prediction"] = False

x = df.iloc[:, 3:]
y = df['skin_cancer_diagnosis']

x_train, x_valtest, y_train, y_valtest = train_test_split(
x, y, test_size = 0.4, random_state = 8008135, stratify = y)

x_val, x_test, y_val, y_test = train_test_split(
x_valtest, y_valtest, test_size = 0.5, random_state = 8008135, stratify = y_valtest)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_val_scaled = scaler.transform(x_val)
x_test_scaled = scaler.transform(x_test)

x = np.arange(0, 1, 0.01)
y = np.arange(1, 11, 0.1)
models = [LogisticRegression(random_state = 8008135, class_weight = {1: i, 0: 1}) for i in y]
for model in models:
    model.fit(x_train_scaled, y_train) 
preds = [model.predict_proba(x_val_scaled)[:, 1] for model in models]
z = np.array([[accuracy_score(y_val, (j >= i).astype(int)) for i in x] for j in preds])

fig = go.Figure()
fig.add_trace(go.Surface(x = x, y = y, z = z))


# In[ ]:




