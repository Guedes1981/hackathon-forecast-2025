# src/evaluate_baselines.py
from pathlib import Path
import time
import numpy as np
import pandas as pd

# ====================== Parâmetros ======================
BASE_DIR     = Path(__file__).resolve().parents[1]
PROC_DIR     = BASE_DIR / "data" / "processed"
REPORT_DIR   = BASE_DIR / "reports"

TRAIN_WEEKS  = list(range(1, 49))   # 1..48
VAL_WEEKS    = [49, 50, 51, 52]     # 49..52
MEAN_WINDOWS = [4, 8]               # comparar janelas 4 e 8
SAVE_DETAILS = False                # True para salvar details (maior I/O)
# =======================================================

def wmape(df_in: pd.DataFrame) -> float:
    den = df_in["y"].abs().sum()
    if den == 0:
        return np.nan
    num = (df_in["y"] - df_in["qtd_prev"]).abs().sum()
    return float(num / den)

def main():
    t0 = time.time()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print(">> Lendo transacoes_2022_diarias.parquet")
    df = pd.read_parquet(
        PROC_DIR / "transacoes_2022_diarias.parquet",
        columns=["data", "pdv", "produto", "quantidade"]
    )
    print("shape diário:", df.shape)

    print(">> Gerando chaves semanais (ISO)")
    df["ano"]    = df["data"].dt.isocalendar().year
    df["semana"] = df["data"].dt.isocalendar().week
    df = df[df["ano"] == 2022]

    print(">> Agregando para semanal")
    df_week = df.groupby(["pdv", "produto", "semana"], as_index=False)["quantidade"].sum()
    df_week["quantidade"] = pd.to_numeric(df_week["quantidade"], errors="coerce").astype("float64")
    print("shape semanal:", df_week.shape)

    print(">> Split treino/validação")
    df_train = df_week[df_week["semana"].isin(TRAIN_WEEKS)].copy()
    df_val   = df_week[df_week["semana"].isin(VAL_WEEKS)].copy().rename(columns={"quantidade": "y"})
    print("treino:", df_train["semana"].min(), "→", df_train["semana"].max(),
          "| validação:", min(VAL_WEEKS), "→", max(VAL_WEEKS))
    print("séries na validação:", df_val.groupby(["pdv","produto"]).ngroups)

    pred_list = []
    cutoff = max(TRAIN_WEEKS)

    # ===== Baselines de média das últimas N semanas =====
    for w in MEAN_WINDOWS:
        low = max(min(TRAIN_WEEKS), cutoff - w + 1)
        print(f">> Baseline mean_last_{w}: semanas usadas no treino = {low}..{cutoff}")
        ref = df_train[(df_train["semana"] >= low) & (df_train["semana"] <= cutoff)]
        m = (ref.groupby(["pdv","produto"], as_index=False)["quantidade"]
               .mean()
               .rename(columns={"quantidade": "qtd_prev"}))
        # prever somente chaves presentes na validação (repete o mesmo valor em 49–52)
        tmp = df_val.merge(m, on=["pdv","produto"], how="left")
        tmp["modelo"] = f"mean_last_{w}"
        pred_list.append(tmp)

    # ===== Baseline naïve sazonal: usa a mesma semana ISO do treino =====
    print(">> Baseline naive_seasonal: mesma semana ISO do treino")
    seasonal = df_train[df_train["semana"].isin(VAL_WEEKS)].rename(columns={"quantidade": "qtd_prev"})
    tmp = df_val.merge(seasonal[["pdv","produto","semana","qtd_prev"]],
                       on=["pdv","produto","semana"], how="left")
    tmp["modelo"] = "naive_seasonal"
    pred_list.append(tmp)

    print(">> Concatenando previsões")
    eval_df = pd.concat(pred_list, ignore_index=True)
    eval_df["qtd_prev"] = eval_df["qtd_prev"].fillna(0).clip(lower=0)
    eval_df["y"]        = eval_df["y"].fillna(0)

    print(">> Calculando métricas (WMAPE)")
    overall = eval_df.groupby(["modelo"]).apply(wmape).reset_index(name="WMAPE_overall")

    per_series = (eval_df.groupby(["modelo","pdv","produto"])
                  .apply(wmape)
                  .reset_index(name="WMAPE_series"))
    summary = (per_series.groupby("modelo")["WMAPE_series"]
               .agg(count="count",
                    p50=lambda s: np.nanpercentile(s, 50),
                    p90=lambda s: np.nanpercentile(s, 90))
               .reset_index())

    # relatórios
    overall_path = REPORT_DIR / "baseline_eval_overall.csv"
    summary_path = REPORT_DIR / "baseline_eval_summary.csv"
    overall.to_csv(overall_path, index=False)
    summary.to_csv(summary_path, index=False)

    if SAVE_DETAILS:
        details_path = REPORT_DIR / "baseline_eval_details.csv"
        eval_df.to_csv(details_path, index=False)
        print("details salvo em:", details_path)

    print("\nRelatórios salvos:")
    print(" -", overall_path)
    print(" -", summary_path)

    print("\n=== Overall WMAPE ===")
    print(overall.to_string(index=False))
    print("\n=== Summary (por série) ===")
    print(summary.to_string(index=False))

    print("\nTempo total: {:.1f}s".format(time.time() - t0))

if __name__ == "__main__":
    main()
