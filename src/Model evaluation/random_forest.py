import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

path = "..\..\data\Metadata_features\metadata_features.csv"
df = pd.read_csv(path)

X = df[['melanoma_color_count', 'hue_variance',
        'saturation_variance', 'value_variance', 'mabrouk_asymmetry_score',
        'avg_asymmetry_score', 'worst_score', 'Polsby-Popper', 'convexity_score',
        'lesion_red_share', 'lesion_green_share', 'lesion_blue_share',
        'lesion_skin_red_diff', 'lesion_skin_green_diff', 'lesion_skin_blue_diff',
        'lesion_skin_rgb_distance']]
y = df["skin_cancer_diagnosis"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


threshold = 0.8
max_allowed = 3
corr = X_train.corr().abs().fillna(0)
counts = (corr > threshold).sum(axis=1) - 1
to_drop = counts[counts > max_allowed].index.tolist()
X_reduced = X_train.drop(columns=to_drop)
label_col = "skin_cancer_diagnosis"
df_rank = X_reduced.copy()
df_rank[label_col] = y_train
corr_with_label = df_rank.corr().abs()[label_col].drop(label_col).fillna(0)
top10 = corr_with_label.sort_values(ascending=False).head(10).index.tolist()


X_final = df[top10]
y_final = df["skin_cancer_diagnosis"]


model = RandomForestClassifier(
    n_estimators=500,
    max_depth=8,
    random_state=42,
    class_weight={0: 1, 1: 1.5}
)
model.fit(X_final, y_final)

all_pred = model.predict(X_final)
all_proba = model.predict_proba(X_final)
df["predicted_label"] = all_pred
df["predicted_probability_cancer"] = all_proba[:, list(model.classes_).index(1)]


df.to_csv("../../data/Metadata_features/metadata_features.csv", index=False)
