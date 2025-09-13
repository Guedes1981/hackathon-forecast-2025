import argparse
import pandas as pd, numpy as np
from tqdm import tqdm
from prophet import Prophet
from common import resolve_project_dir

parser = argparse.ArgumentParser()
parser.add_argument("--top_n", type=int, default=200)
args = parser.parse_args()

PROJECT_DIR = resolve_project_dir()
wk_path = PROJECT_DIR / "data" / "processed" / "train_weekly_splits.parquet"
out_path = PROJECT_DIR / "data" / "processed" / "forecast_ensemble_jan2023.parquet"

wk = pd.read_parquet(wk_path)
wk["semana"] = pd.to_datetime(wk["semana"]).dt.normalize()
wk["y"] = wk["quantidade"].astype(float)
key = ["pdv","produto"]

forecast_weeks = pd.to_datetime(["2023-01-02","2023-01-09","2023-01-16","2023-01-23"])
cutoff = pd.Timestamp("2022-12-26")

# Top-N por volume no treino
train_mask = wk["split"].eq("train")
sum_by_pair = (wk.loc[train_mask].groupby(key, sort=False)["y"].sum().rename("sum_y").reset_index())
top_pairs = sum_by_pair.sort_values("sum_y", ascending=False).head(args.top_n)

# Prophet nas Top-N
prophet_rows = []
for _, row in tqdm(top_pairs.iterrows(), total=len(top_pairs)):
    pdv, produto = str(row["pdv"]), str(row["produto"])
    sub = wk[(wk["pdv"]==pdv) & (wk["produto"]==produto)].copy()
    train = (sub[sub["semana"] <= cutoff][["semana","y"]]
             .rename(columns={"semana":"ds","y":"y"})
             .sort_values("ds"))
    if len(train) < 8:
        continue

    m = Prophet(
        growth="linear",
        weekly_seasonality=True,
        yearly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="additive",
        changepoint_prior_scale=0.5,
        interval_width=0.8
    )
    m.fit(train)
    future = pd.DataFrame({"ds": forecast_weeks})
    fcst = m.predict(future)[["ds","yhat"]]
    fcst["pdv"] = pdv
    fcst["produto"] = produto
    prophet_rows.append(fcst)

if prophet_rows:
    prophet_fc = pd.concat(prophet_rows, ignore_index=True).rename(columns={"ds":"semana","yhat":"quantidade"})
    prophet_fc["quantidade"] = prophet_fc["quantidade"].clip(lower=0)
else:
    prophet_fc = pd.DataFrame(columns=["semana","pdv","produto","quantidade"])

prophet_pairs = set(zip(prophet_fc["pdv"], prophet_fc["produto"]))

# MA4 para cauda longa (último MA4 até o cutoff replicado nas 4 semanas)
wk_sorted = wk.sort_values(key+["semana"]).copy()
wk_sorted["ma4"] = wk_sorted.groupby(key)["y"].rolling(4, min_periods=1).mean().reset_index(level=key, drop=True)

last_train = wk_sorted[wk_sorted["semana"] <= cutoff]
last_ma4 = (last_train.groupby(key, as_index=False)
            .apply(lambda s: s.iloc[-1][["ma4"]])
            .reset_index(drop=True))
last_ma4["ma4"] = last_ma4["ma4"].fillna(0).clip(lower=0)

all_pairs = set(zip(wk[key[0]].astype(str), wk[key[1]].astype(str)))
tail_pairs = all_pairs.difference(prophet_pairs)
tail_df = pd.DataFrame(list(tail_pairs), columns=key)

tail_base = (tail_df.merge(last_ma4, on=key, how="left").fillna({"ma4":0}))
tail_base = tail_base.assign(_k=1).merge(
    pd.DataFrame({"semana": forecast_weeks, "_k":[1,1,1,1]}),
    on="_k", how="left").drop(columns="_k")
tail_base = tail_base.rename(columns={"ma4":"quantidade"})

# Ensemble
ens = pd.concat([
    prophet_fc[["semana","pdv","produto","quantidade"]],
    tail_base[["semana","pdv","produto","quantidade"]]
], ignore_index=True)

ens["semana"] = pd.to_datetime(ens["semana"]).dt.normalize()
ens["pdv"] = ens["pdv"].astype("string")
ens["produto"] = ens["produto"].astype("string")
ens["quantidade"] = ens["quantidade"].clip(lower=0).round().astype(int)
ens = ens.sort_values(["semana","pdv","produto"]).reset_index(drop=True)

ens.to_parquet(out_path, index=False)
print(out_path)
