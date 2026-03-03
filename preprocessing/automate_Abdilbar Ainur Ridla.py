import os
import pandas as pd

TARGET = "Survived"

def preprocess(input_path: str, output_path: str) -> str:
    df = pd.read_csv(input_path)

    df2 = df.copy()
    drop_cols = ["Cabin", "Ticket", "Name"]
    df2 = df2.drop(columns=[c for c in drop_cols if c in df2.columns])

    df2 = df2.drop_duplicates()

    for c in df2.columns:
        if c == TARGET:
            continue
        if df2[c].dtype == "O":
            df2[c] = df2[c].fillna(df2[c].mode()[0])
        else:
            df2[c] = df2[c].fillna(df2[c].median())

    df2 = pd.get_dummies(df2, columns=["Sex","Embarked"], drop_first=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df2.to_csv(output_path, index=False)
    return output_path

if __name__ == "__main__":
    in_path = "namadataset_raw/data.csv"
    out_path = "preprocessing/namadataset_preprocessing/data_processed.csv"
    print("Saved:", preprocess(in_path, out_path))