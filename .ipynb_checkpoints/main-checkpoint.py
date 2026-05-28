import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
import joblib



names = ("features.csv", "biggest_features.csv", "top3_features.csv", "no_hair_features.csv", 
         "biggest_no_hair_features.csv", "top3_no_hair_features.csv")


path = f"data/features/"
dfs = [pd.read_csv(path + x) for x in names]

def __main__(load_model):
    for i in range(len(dfs)):
        
        df = dfs[i]
        output_name = names[i]
    
        X = df[['melanoma_color_count', 'hue_variance',
                'saturation_variance', 'value_variance', 'mabrouk_asymmetry_score',
                'avg_asymmetry_score', 'worst_score', 'Polsby-Popper', 'convexity_score',
                'lesion_red_share', 'lesion_green_share', 'lesion_blue_share',
                'lesion_skin_red_diff', 'lesion_skin_green_diff', 'lesion_skin_blue_diff',
                'lesion_skin_rgb_distance']]
        y = df["skin_cancer_diagnosis"]
    
        if load_model == False:
            patient_labels = df.groupby("patient_id")["skin_cancer_diagnosis"].max()

            
            threshold = 0.8
            max_allowed = 3
            corr = X.corr().abs().fillna(0)
            counts = (corr > threshold).sum(axis=1) - 1
            to_drop = counts[counts > max_allowed].index.tolist()
            X_reduced = X.drop(columns=to_drop)
            label_col = "skin_cancer_diagnosis"
            df_rank = X_reduced.copy()
            df_rank[label_col] = y
            corr_with_label = df_rank.corr().abs()[label_col].drop(label_col).fillna(0)
            top10 = corr_with_label.sort_values(ascending=False).head(10).index.tolist()
        
            X = X[top10]
        
            groups = df.loc[X.index, "patient_id"]
        
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
                        X,
                        y,
                        cv=cv,
                        groups=groups,
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
            model.fit(X, y)
    
            joblib.dump(model, "results/models/" + output_name[0:-4] + "_model.pkl")
            with open("results/models/" + output_name[0:-4] + "_model_top10.txt", "w") as file:
                file.write('\n'.join(top10))
                
            
    
        else:
            model = joblib.load("results/models/" + output_name[0:-4] + "_model.pkl")
            with open("results/models/" + output_name[0:-4] + "_model_top10.txt", "r") as file:
                top10 = file.read().splitlines()
            all_pred = model.predict(X.loc[:, top10])
            all_proba = model.predict_proba(X.loc[:, top10])
        
            df_output = df[["img_id", "patient_id", "diagnostic", "skin_cancer_diagnosis"]].copy()
        
            df_output["predicted_label"] = all_pred
            df_output["predicted_probability_cancer"] = all_proba[:, list(model.classes_).index(1)]
        
        
            df_output.to_csv("results/predictions/" + output_name, index=False)
        print(str(i+1) + "/6 finished")
    
__main__(True)
    
    
