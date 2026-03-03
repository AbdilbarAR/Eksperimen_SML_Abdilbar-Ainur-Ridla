import os
import pandas as pd

TARGET = "Survived"

def preprocess(input_path: str, output_path: str) -> str:
    df = pd.read_csv(input_path)

    df2 = df.copy()

    # Drop kolom yang tidak dipakai (kalau ada)
    drop_cols = ["Cabin", "Ticket", "Name"]
    df2 = df2.drop(columns=[c for c in drop_cols if c in df2.columns], errors="ignore")

    # Drop duplicate rows
    df2 = df2.drop_duplicates()

    # --- Imputation yang aman ---
    # 1) Kolom numerik: isi NaN dengan median
    num_cols = df2.select_dtypes(include=["number"]).columns.tolist()
    for c in num_cols:
        if c == TARGET:
            continue
        med = df2[c].median()
        # kalau semua NaN -> median jadi NaN, isi 0 biar tidak kosong
        if pd.isna(med):
            med = 0
        df2[c] = df2[c].fillna(med)

    # 2) Kolom non-numerik (object/string/bool/category/datetime): isi NaN dengan mode / fallback
    non_num_cols = [c for c in df2.columns if c not in num_cols and c != TARGET]
    for c in non_num_cols:
        s = df2[c]
        # pastikan mode aman walau kosong
        mode_vals = s.mode(dropna=True)
        fill_val = mode_vals.iloc[0] if len(mode_vals) > 0 else "UNKNOWN"
        df2[c] = s.fillna(fill_val)

    # --- One-hot encoding (hanya kalau kolomnya ada) ---
    dummy_cols = [c for c in ["Sex", "Embarked"] if c in df2.columns]
    if dummy_cols:
        df2 = pd.get_dummies(df2, columns=dummy_cols, drop_first=True)

    # Save
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df2.to_csv(output_path, index=False)
    return output_path

if __name__ == "__main__":
    in_path = "namadataset_raw/data.csv"
    out_path = "preprocessing/namadataset_preprocessing/data_processed.csv"
    print("Saved:", preprocess(in_path, out_path))