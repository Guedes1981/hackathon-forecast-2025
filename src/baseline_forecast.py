# src/baseline_forecast.py
from pathlib import Path
import pandas as pd

# === Parâmetros ===
N_SEMANAS_MEDIA = 8       # janelas de média
N_SEMANAS_JAN   = 5       # gerar 4 ou 5 semanas (ajuste aqui)

base_dir = Path(__file__).resolve().parents[1]
data_proc = base_dir / "data" / "processed"

# === Carregar transações diárias ===
df = pd.read_parquet(data_proc / "transacoes_2022_diarias.parquet")

# === Agregar por semana ISO ===
df["ano"] = df["data"].dt.isocalendar().year
df["semana"] = df["data"].dt.isocalendar().week

df_sem = (df.query("ano == 2022")
            .groupby(["pdv","produto","semana"], as_index=False)["quantidade"]
            .sum())

# === Baseline: média móvel das últimas N semanas ===
last_weeks = df_sem.query("semana > (52 - @N_SEMANAS_MEDIA)")
df_baseline = (last_weeks
               .groupby(["pdv","produto"], as_index=False)["quantidade"]
               .mean()
               .rename(columns={"quantidade":"qtd_prev"}))

# === Replicar previsão para semanas de janeiro/2023 ===
df_forecast = pd.DataFrame(
    [(w,p,s,q) for w in range(1, N_SEMANAS_JAN+1)
                for _,(p,s,q) in df_baseline.iterrows()],
    columns=["semana","pdv","produto","qtd_prev"]
)

# === Ajustes finais ===
df_forecast["quantidade"] = (df_forecast["qtd_prev"]
                             .round()
                             .clip(lower=0)
                             .astype(int))
df_forecast = df_forecast.drop(columns=["qtd_prev"])

# === Exportar ===
out_csv = data_proc / "baseline_forecast.csv"
out_parquet = data_proc / "baseline_forecast.parquet"

df_forecast.to_csv(out_csv, sep=";", index=False, encoding="utf-8")
df_forecast.to_parquet(out_parquet, index=False)

print("Salvos:")
print(" -", out_csv)
print(" -", out_parquet)
print("shape :", df_forecast.shape)
print("colunas:", list(df_forecast.columns))
print(df_forecast.head(10).to_string(index=False))
