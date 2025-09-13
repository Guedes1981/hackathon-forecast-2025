import numpy as np
import pandas as pd
from common import resolve_project_dir

def wmape(y_true, y_pred):
    denom = np.sum(np.abs(y_true))
    return np.nan if denom == 0 else np.sum(np.abs(y_true - y_pred)) / denom

PROJECT_DIR = resolve_project_dir()
in_path  = PROJECT_DIR / "data" / "processed" / "train_weekly_splits.parquet"
pred_out = PROJECT_DIR / "data" / "processed" / "baseline_preds.parquet"
met_out  = PROJECT_DIR / "reports" / "_baseline_metrics.csv"

wk = pd.read_parquet(in_path)
wk["semana"] = pd.to_datetime(wk["semana"]).dt.normalize()
wk = wk.sort_values(["pdv","produto","semana"]).reset_index(drop=True)
key = ["pdv","produto"]

wk["y"]     = wk["quantidade"].astype(float)
wk["y_lag1"]= wk.groupby(key)["y"].shift(1)
wk["y_lag4"]= wk.groupby(key)["y"].shift(4)
wk["ma4"]   = wk.groupby(key)["y"].rolling(4, min_periods=1).mean().reset_index(level=key, drop=True)
wk["ma8"]   = wk.groupby(key)["y"].rolling(8, min_periods=1).mean().reset_index(level=key, drop=True)

baselines = {"lastweek":"y_lag1","seasonal4":"y_lag4","ma4":"ma4","ma8":"ma8"}

metrics, preds = [], []
for name, col in baselines.items():
    for split in ["val8","val4"]:
        sub = wk[wk["split"]==split].copy()
        yhat = sub[col].fillna(0.0).values
        score = wmape(sub["y"].values, yhat)
        metrics.append({"model":name,"split":split,"wmape":float(score)})
        pf = sub[["semana","pdv","produto","y"]].copy()
        pf["quantidade_hat"] = yhat
        pf["model"] = name
        pf["split"] = split
        preds.append(pf)

pred = pd.concat(preds, ignore_index=True)
pred.to_parquet(pred_out, index=False)
pd.DataFrame(metrics).to_csv(met_out, index=False)
print(pred_out)
print(met_out)
