# src/prepare_transacoes_diarias.py
from pathlib import Path
import pandas as pd

base_dir = Path(__file__).resolve().parents[1]
data_proc = base_dir / "data" / "processed"
in_path  = data_proc / "transacoes_2022.parquet"
out_path = data_proc / "transacoes_2022_diarias.parquet"

df = pd.read_parquet(in_path)

n_before = len(df)
df = (df
      .groupby(["data","pdv","produto"], as_index=False, sort=False)["quantidade"]
      .sum())
# tipos
df["pdv"] = df["pdv"].astype("string")
df["produto"] = df["produto"].astype("string")
df["data"] = pd.to_datetime(df["data"]).dt.normalize()
df["quantidade"] = pd.to_numeric(df["quantidade"], errors="coerce").astype("Float64")

n_after = len(df)
print("entrada :", in_path)
print("linhas antes :", n_before)
print("linhas depois:", n_after)
print("redução      :", n_before - n_after)
print("datas       :", df["data"].min().date(), "→", df["data"].max().date())
print("pdvs únicos :", df["pdv"].nunique())
print("skus únicos :", df["produto"].nunique())

out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(out_path, index=False)
print("salvo em    :", out_path)
print(df.head(5).to_string(index=False))
