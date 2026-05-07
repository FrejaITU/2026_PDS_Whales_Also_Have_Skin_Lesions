import pandas as pd

df = pd.read_csv("../data/hair_coverage.csv")

print(df.sort_values("hair_coverage").head(10))
print(df.sort_values("hair_coverage").tail(10))

print(
    df.iloc[
        (df["hair_coverage"] - df["hair_coverage"].median())
        .abs()
        .argsort()[:10]
    ]
)