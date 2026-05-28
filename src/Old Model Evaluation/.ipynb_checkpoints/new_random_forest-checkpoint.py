import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import joblib

names = ("features.csv", "biggest_features.csv", "top3_features.csv", "no_hair_features.csv", 
         "biggest_no_hair_features.csv", "top3_no_hair_features.csv")


path = f"../../data/features/"
dfs = [pd.read_csv(path + x) for x in names]

def run(df, output_name):
    X = df[['melanoma_color_count', 'hue_variance',
            'saturation_variance', 'value_variance', 'mabrouk_asymmetry_score',
            'avg_asymmetry_score', 'worst_score', 'Polsby-Popper', 'convexity_score',
            'lesion_red_share', 'lesion_green_share', 'lesion_blue_share',
            'lesion_skin_red_diff', 'lesion_skin_green_diff', 'lesion_skin_blue_diff',
            'lesion_skin_rgb_distance']]
    y = df["skin_cancer_diagnosis"]


    patient_labels = df.groupby("patient_id")["skin_cancer_diagnosis"].max()

    patients = patient_labels.index
    labels = patient_labels.values

    train_patients, test_patients = train_test_split(
        patients,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    train_mask = df["patient_id"].isin(train_patients)
    test_mask = df["patient_id"].isin(test_patients)

    X_train = X[train_mask]
    y_train = y[train_mask]

    X_test = X[test_mask]
    y_test = y[test_mask]

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

    print(f"Top 10 features for {output_name}: {top10}")

    X_train = X_train[top10]
    X_test = X_test[top10]

    groups_train = df.loc[X_train.index, "patient_id"]

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    depths = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20]
    estimators = [10, 50, 100, 200, 300, 500]

    results = []

    for n in estimators:
        for depth in depths:
            rf = RandomForestClassifier(
                n_estimators=n,
                max_depth=depth,
                random_state=42,
                class_weight={0: 1, 1: 2}
            )

            scores = cross_val_score(
                rf,
                X_train,
                y_train,
                cv=cv,
                groups=groups_train,
                scoring='roc_auc',
                n_jobs=-1
            )

            mean_score = scores.mean()


            results.append((
                n,
                depth,
                mean_score,
            ))


    top1 = sorted(results, key=lambda x: x[2])[-1]

    n_estimators = top1[0]
    depth= top1[1]



    model = RandomForestClassifier(
        n_estimators= n_estimators,
        max_depth= depth,
        random_state=42,
        class_weight={0: 1, 1: 2}
    )
    model.fit(X_train, y_train)

    #joblib.dump(model, "results/models/" + output_name[0:-4] + "_model.pkl")

    all_pred = model.predict(X_test)
    all_proba = model.predict_proba(X_test)

    df_output = df.loc[test_mask, ["img_id", "patient_id", "diagnostic", "skin_cancer_diagnosis"]].copy()

    df_output["predicted_label"] = all_pred
    df_output["predicted_probability_cancer"] = all_proba[:, list(model.classes_).index(1)]

    print(f"Test AUC of {output_name}: {roc_auc_score(y_test, df_output["predicted_probability_cancer"])}")
    print(f"Test Acc of {output_name}: {accuracy_score(y_test, df_output["predicted_label"])}")
    


    #df_output.to_csv("results/predictions/" + output_name, index=False)

for i in range(len(dfs)):
    run(dfs[i], names[i])
    print(str(i+1) + "/6 finished")

