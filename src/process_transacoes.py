# src/process_transacoes.py
from pathlib import Path
import sys
import pyarrow.parquet as pq
import pandas as pd

def main():
    base_dir = Path(__file__).resolve().parents[1]
    data_raw  = base_dir / "data" / "raw"
    data_proc = base_dir / "data" / "processed"
    out_path  = data_proc / "transacoes_2022.parquet"

    print("BASE_DIR:", base_dir)
    print("RAW_DIR :", data_raw)
    print("PROC_DIR:", data_proc)

    raw_files = sorted(data_raw.glob("*.parquet"))
    if not raw_files:
        raise FileNotFoundError(f"Nenhum .parquet encontrado em {data_raw}")

    # Heurística: transações devem ter quantidade + data + ids internos de pdv/produto
    MUST_HAVE_ANY_DATE = {"transaction_date", "reference_date"}
    MUST_HAVE_ALL      = {"quantity", "internal_store_id", "internal_product_id"}

    def lower_cols(pfile: pq.ParquetFile) -> set:
        return {f.name.lower() for f in pfile.schema_arrow}

    schemas = {}
    trx_path = None
    for f in raw_files:
        pfile = pq.ParquetFile(f)
        cols = lower_cols(pfile)
        schemas[f.name] = cols
        has_req = MUST_HAVE_ALL.issubset(cols)
        has_date = len(MUST_HAVE_ANY_DATE & cols) > 0
        if has_req and has_date:
            trx_path = f
            break

    if trx_path is None:
        details = "\n".join([f"- {k}: {sorted(list(v))}" for k, v in schemas.items()])
        raise RuntimeError("Arquivo de transações não identificado automaticamente. Schemas detectados:\n" + details)

    print("Arquivo de transações identificado:", trx_path.name)

    # Determinar coluna de data preferencial
    trx_cols = schemas[trx_path.name]
    date_col = "transaction_date" if "transaction_date" in trx_cols else "reference_date"

    # Colunas mínimas para leitura
    read_cols = ["internal_store_id", "internal_product_id", date_col, "quantity"]
    print("Colunas selecionadas para leitura:", read_cols)

    # Leitura por row groups (baixo uso de memória)
    pfile = pq.ParquetFile(trx_path)
    dfs = []
    print("Row groups:", pfile.num_row_groups)
    for rg in range(pfile.num_row_groups):
        tbl = pfile.read_row_group(rg, columns=read_cols)
        dfs.append(tbl.to_pandas())

    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=read_cols)

    # Normalização de nomes e tipos
    df = df.rename(columns={
        "internal_store_id":   "pdv",
        "internal_product_id": "produto",
        date_col:              "data",
        "quantity":            "quantidade",
    })

    # Tipos
    df["pdv"]      = df["pdv"].astype("string").str.strip()
    df["produto"]  = df["produto"].astype("string").str.strip()
    df["quantidade"] = pd.to_numeric(df["quantidade"], errors="coerce").astype("Float64")

    # Datas: para análises semanais, manter como data normalizada (meia-noite)
    df["data"] = pd.to_datetime(df["data"], errors="coerce")
    df["data"] = df["data"].dt.normalize()

    # Remover linhas inválidas (campos essenciais nulos)
    df = df.dropna(subset=["pdv", "produto", "data", "quantidade"]).reset_index(drop=True)

    # Persistência
    data_proc.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    # Logs essenciais
    print("Salvo em:", out_path)
    print("shape :", df.shape)
    print("colunas:", list(df.columns))
    print(df.head(5).to_string(index=False))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[ERRO]", type(e).__name__, "-", e, file=sys.stderr)
        sys.exit(1)
