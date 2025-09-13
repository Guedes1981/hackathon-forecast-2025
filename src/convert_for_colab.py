# src/convert_for_colab.py
from pathlib import Path
import pandas as pd

BASE = Path(r"C:\Users\guede\OneDrive\Documentos\Hackathon\Bigdata\Arquivos\hackathon-forecast-2025")
PROC = BASE / "data" / "processed"

in_parquet  = PROC / "transacoes_2022_diarias.parquet"
out_feather = PROC / "transacoes_2022_diarias.feather"

print("lendo:", in_parquet)
df = pd.read_parquet(in_parquet, columns=["data","pdv","produto","quantidade"])
print("shape:", df.shape, "| colunas:", list(df.columns))

print("salvando Feather:", out_feather)
df.to_feather(out_feather, compression="zstd")  # compacto e rápido

print("ok:", out_feather, "| tamanho (MB):", round(out_feather.stat().st_size/1024/1024,2))
