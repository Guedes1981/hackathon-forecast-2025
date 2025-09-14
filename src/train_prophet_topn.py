import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from prophet import Prophet
from common import resolve_project_dir

def wmape(y_true, y_pred):
    denom = np.sum(np.abs(y_true))
    return np.nan if denom == 0 else np.sum(np.abs(y_true - y_pred)) / denom

parser = argparse.ArgumentParser()

parser.add_argument("--predict_jan2023", action="store_true", help="Gera previsões de produção para Jan/2023 (02–23/01)")

parser.add_argument("--predict-jan2023", dest="predict_jan2023", action="store_true", help="Alias para --predict_jan2023")
parser.add_argument("--top_n", type=int, default=200)
args = parser.parse_known_args()[0]

PROJECT_DIR = resolve_project_dir()
wk = pd.read_parquet(PROJECT_DIR / "data" / "processed" / "train_weekly_splits.parquet")
wk["semana"] = pd.to_datetime(wk["semana"]).dt.normalize()
wk["y"] = wk["quantidade"].astype(float)
key = ["pdv","produto"]

train_mask = wk["split"].eq("train")
sum_by_pair = (wk.loc[train_mask]
                 .groupby(key, sort=False)["y"]
                 .sum().rename("sum_y").reset_index())
top_pairs = sum_by_pair.sort_values("sum_y", ascending=False).head(args.top_n)

weeks_val4 = sorted(wk.loc[wk["split"].eq("val4"), "semana"].unique())
cutoff = weeks_val4[0] - pd.Timedelta(weeks=1)

pred_rows = []
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

    future = pd.DataFrame({"ds": weeks_val4})
    fcst = m.predict(future)[["ds","yhat"]]
    fcst["pdv"] = pdv
    fcst["produto"] = produto

    truth = (sub[sub["semana"].isin(weeks_val4)][["semana","y"]]
             .rename(columns={"semana":"ds"}))
    merged = truth.merge(fcst, on="ds", how="left")
    merged["yhat"] = merged["yhat"].clip(lower=0)
    merged["model"] = "prophet_topN"
    pred_rows.append(merged.assign(pdv=pdv, produto=produto))

preds = (pd.concat(pred_rows, ignore_index=True)
         if pred_rows else
         pd.DataFrame(columns=["ds","pdv","produto","y","yhat","model"]))

preds = preds.rename(columns={"ds":"semana"}).sort_values(["pdv","produto","semana"])
preds.to_parquet(PROJECT_DIR / "data" / "processed" / "prophet_topN_val4_preds.parquet", index=False)

score = wmape(preds["y"].values, preds["yhat"].values) if len(preds) else np.nan
(pd.DataFrame([{"model":"prophet_topN","split":"val4","wmape":float(score)}])
   .to_csv(PROJECT_DIR / "reports" / "_prophet_val4_metrics.csv", index=False))
print(PROJECT_DIR / "data" / "processed" / "prophet_topN_val4_preds.parquet")
print(PROJECT_DIR / "reports" / "_prophet_val4_metrics.csv")

def _save_prophet_jan_from_val4(out_parquet, jan_ini="2023-01-02", jan_fim="2023-01-23"):
    import os, pandas as pd
    val4 = "data/processed/prophet_topN_val4_preds.parquet"
    if not os.path.exists(val4):
        raise FileNotFoundError(f"Arquivo não encontrado: {val4}")
    df = pd.read_parquet(val4)
    # Mantém apenas as colunas esperadas e a janela de JAN/2023
    cols = [c for c in ["sku_id","pdv_id","ds","yhat"] if c in df.columns]
    df = df.loc[(df["ds"] >= jan_ini) & (df["ds"] <= jan_fim), cols]
    df.to_parquet(out_parquet, index=False)
    print(f"OK - prophet Jan salvo: {out_parquet} | linhas: {len(df)}")
