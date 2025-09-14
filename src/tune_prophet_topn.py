import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from prophet import Prophet
from common import resolve_project_dir

def wmape(y_true, y_pred):
    denom = np.abs(y_true).sum()
    return float(np.abs(y_true - y_pred).sum() / (denom if denom != 0 else 1.0))

def prepare_inputs(PROJECT_DIR):
    wk = pd.read_parquet(PROJECT_DIR / "data" / "processed" / "train_weekly_splits.parquet")
    wk["semana"] = pd.to_datetime(wk["semana"]).dt.normalize()
    wk["y"] = wk["quantidade"].astype(float)
    key = ["pdv","produto"]
    cutoff = pd.Timestamp("2022-12-26")
    val4_weeks = pd.to_datetime(["2022-12-05","2022-12-12","2022-12-19","2022-12-26"])
    # rolling MA4 (para fallback)
    wk = wk.sort_values(key+["semana"])
    wk["ma4"] = wk.groupby(key)["y"].rolling(4, min_periods=1).mean().reset_index(level=key, drop=True)
    return wk, key, cutoff, val4_weeks

def run_prophet_for_pairs(wk, key, cutoff, pairs, cps):
    rows = []
    for pdv, produto in tqdm(pairs, leave=False):
        sub = wk[(wk["pdv"]==pdv) & (wk["produto"]==produto)]
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
            changepoint_prior_scale=cps,
            interval_width=0.8,
        )
        m.fit(train)
        # prever em val4 para avaliação
        future = pd.DataFrame({"ds": pd.to_datetime(["2022-12-05","2022-12-12","2022-12-19","2022-12-26"])})
        fcst = m.predict(future)[["ds","yhat"]]
        fcst["pdv"] = pdv
        fcst["produto"] = produto
        rows.append(fcst.rename(columns={"ds":"semana","yhat":"yhat"}))
    if rows:
        out = pd.concat(rows, ignore_index=True)
        out["semana"] = pd.to_datetime(out["semana"]).dt.normalize()
    else:
        out = pd.DataFrame(columns=["semana","pdv","produto","yhat"])
    return out

def evaluate_config(wk, key, cutoff, val4_weeks, top_n, cps):
    # Top-N por volume no treino
    train_mask = wk["semana"] <= cutoff
    sum_by_pair = (wk.loc[train_mask].groupby(key, sort=False)["y"].sum().rename("sum_y").reset_index())
    top_pairs = (sum_by_pair.sort_values("sum_y", ascending=False).head(top_n))
    top_pairs_list = list(map(tuple, top_pairs[key].astype(str).to_records(index=False)))

    # Prophet em Top-N
    prophet_fc = run_prophet_for_pairs(wk, key, cutoff, top_pairs_list, cps)

    # Construir y_true em val4
    val4 = wk[wk["semana"].isin(val4_weeks)][["semana"]+key+["y","ma4"]].copy()
    val4[key[0]] = val4[key[0]].astype(str)
    val4[key[1]] = val4[key[1]].astype(str)

    # Merge com previsões Prophet (onde houver)
    df = val4.merge(prophet_fc.rename(columns={"yhat":"y_pred"}), on=["semana"]+key, how="left")

    # Fallback: MA4 quando y_pred for NaN
    df["y_pred"] = df["y_pred"].fillna(df["ma4"]).fillna(0)

    score = wmape(df["y"].values, df["y_pred"].values)
    return score, len(prophet_fc[key[0]].unique()), len(prophet_fc)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topn_list", type=str, default="100,200,300")
    parser.add_argument("--cps_list", type=str, default="0.3,0.5,0.8")
    args = parser.parse_args()

    PROJECT_DIR = resolve_project_dir()
    wk, key, cutoff, val4_weeks = prepare_inputs(PROJECT_DIR)

    topn_vals = [int(x) for x in args.topn_list.split(",") if x.strip()]
    cps_vals = [float(x) for x in args.cps_list.split(",") if x.strip()]

    records = []
    for top_n in topn_vals:
        for cps in cps_vals:
            score, n_pairs, n_rows = evaluate_config(wk, key, cutoff, val4_weeks, top_n, cps)
            records.append({
                "top_n": top_n,
                "changepoint_prior_scale": cps,
                "wmape_val4": score,
                "n_pairs_predicted": n_pairs,
                "rows_pred": n_rows,
            })
            print(f"[top_n={top_n:>3} | cps={cps:>3}] WMAPE(val4)={score:.6f} | pairs={n_pairs} | rows={n_rows}")

    res = pd.DataFrame(records).sort_values("wmape_val4")
    out_csv = PROJECT_DIR / "reports" / "_tuning_results.csv"
    res.to_csv(out_csv, index=False)

    best = res.iloc[0].to_dict() if not res.empty else {}
    (PROJECT_DIR / "reports" / "_tuning_best.json").write_text(json.dumps(best, indent=2), encoding="utf-8")

    print(">> resultados:", out_csv)
    print(">> melhor:", best)

if __name__ == "__main__":
    main()
