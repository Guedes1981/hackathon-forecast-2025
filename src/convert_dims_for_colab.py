# src/convert_dims_for_colab.py
from pathlib import Path
import pandas as pd

BASE = Path(r"C:\Users\guede\OneDrive\Documentos\Hackathon\Bigdata\Arquivos\hackathon-forecast-2025")
PROC = BASE / "data" / "processed"

pairs = [
    ("produtos.parquet", "produtos.feather"),
    ("pdvs.parquet",     "pdvs.feather"),
]

for p_in, f_out in pairs:
    ppath = PROC / p_in
    fpath = PROC / f_out
    print("lendo:", ppath)
    df = pd.read_parquet(ppath)
    print("shape:", df.shape, "| colunas:", list(df.columns))
    print("salvando:", fpath)
    df.to_feather(fpath, compression="zstd")
    print("ok:", fpath.name)
