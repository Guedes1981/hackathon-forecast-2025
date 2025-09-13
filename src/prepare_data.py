import pandas as pd
from common import resolve_project_dir

PROJECT_DIR = resolve_project_dir()
src = PROJECT_DIR / "data" / "processed" / "df_all.long.parquet"
out = PROJECT_DIR / "data" / "processed" / "train_weekly.parquet"

df = pd.read_parquet(src, columns=["ds","id","y"])
df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
df["y"]  = pd.to_numeric(df["y"], errors="coerce").fillna(0)
df = df.dropna(subset=["ds","id"])

spl = df["id"].astype(str).str.split("|", n=1, expand=True)
df["pdv"] = spl[0]
df["produto"] = spl[1]

df = df[df["ds"].dt.year == 2022]
sem = df["ds"].dt.to_period("W-MON")
df["semana"] = sem.dt.start_time.dt.normalize()

wk = (df.groupby(["semana","pdv","produto"], as_index=False, sort=False)["y"]
        .sum(min_count=1)
        .rename(columns={"y":"quantidade"}))

wk = wk.sort_values(["semana","pdv","produto"]).reset_index(drop=True)
wk.to_parquet(out, index=False)
print(out)
