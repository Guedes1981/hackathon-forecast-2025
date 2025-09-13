# src/eda_basica.py
from pathlib import Path
import pandas as pd

def main():
    base_dir = Path(__file__).resolve().parents[1]
    data_proc = base_dir / "data" / "processed"

    # Carregar dados processados
    df_trx = pd.read_parquet(data_proc / "transacoes_2022.parquet")
    df_pdv = pd.read_parquet(data_proc / "pdvs.parquet")
    df_prod = pd.read_parquet(data_proc / "produtos.parquet")

    print("=== Shapes ===")
    print("Transações :", df_trx.shape)
    print("PDVs       :", df_pdv.shape)
    print("Produtos   :", df_prod.shape)

    print("\n=== Nulos ===")
    print(df_trx.isna().mean().round(4))
    print(df_pdv.isna().mean().round(4))
    print(df_prod.isna().mean().round(4))

    print("\n=== Duplicatas (data,pdv,produto) em Transações ===")
    dup = df_trx.duplicated(subset=["data","pdv","produto"]).sum()
    print("Duplicatas:", dup)

    print("\n=== Transações por semana (2022) ===")
    df_trx["semana"] = df_trx["data"].dt.isocalendar().week
    df_trx["ano"] = df_trx["data"].dt.year
    sem = (df_trx
           .query("ano == 2022")
           .groupby("semana")
           .agg(qtd_trans=("quantidade","count"),
                qtd_pdv=("pdv","nunique"),
                qtd_produto=("produto","nunique"))
           .reset_index()
           .sort_values("semana"))
    print(sem.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
