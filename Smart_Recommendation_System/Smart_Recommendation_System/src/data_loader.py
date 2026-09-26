from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = [
    "id","title","genre","keywords","description","director","cast",
    "rating","popularity"
]

def load_dataset(path=None):
    if path is None:
        path = Path(__file__).resolve().parent.parent / "data" / "movies.csv"
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    for c in ["title","genre","keywords","description","director","cast"]:
        df[c] = df[c].fillna("").astype(str)
    for c in ["rating","popularity"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df
