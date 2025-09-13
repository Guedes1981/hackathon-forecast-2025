import pandas as pd
from common import resolve_project_dir

PROJECT_DIR = resolve_project_dir()
wk_path = PROJECT_DIR / "data" / "processed" / "train_weekly.parquet"
out_path = PROJECT_DIR / "data" / "processed" / "train_weekly_splits.parquet"

wk = pd.read_parquet(wk_path)
wk["semana"] = pd.to_datetime(wk["semana"]).dt.normalize()

weeks_2022 = sorted(wk.loc[wk["semana"].dt.year == 2022, "semana"].unique())
val4 = weeks_2022[-4:] if len(weeks_2022) >= 4 else weeks_2022
val8 = weeks_2022[-8:] if len(weeks_2022) >= 8 else weeks_2022

def label_split(d):
    if d in val4:
        return "val4"
    if d in val8:
        return "val8"
    return "train"

wk["split"] = wk["semana"].map(label_split)
wk.to_parquet(out_path, index=False)
print(out_path)
