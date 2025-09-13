# src/process_pdvs.py
from pathlib import Path
import sys
import pyarrow.parquet as pq
import pandas as pd

def main():
    base_dir = Path(__file__).resolve().parents[1]
    data_raw  = base_dir / "data" / "raw"
    data_proc = base_dir / "data" / "processed"
    out_path  = data_proc / "pdvs.parquet"

    print("BASE_DIR:", base_dir)
    print("RAW_DIR :", data_raw)
    print("PROC_DIR:", data_proc)

    raw_files = sorted(data_raw.glob("*.parquet"))
    if not raw_files:
        raise FileNotFoundError(f"Nenhum .parquet encontrado em {data_raw}")

    # Heurística: PDVs devem conter 'pdv' e não conter colunas de transações
    MUST_HAVE = {"pdv"}
    FORBIDDEN = {"transaction_date", "reference_date", "quantity", "internal_store_id", "internal_product_id"}

    def lower_cols(pfile: pq.ParquetFile) -> set:
        return {f.name.lower() for f in pfile.schema_arrow}

    schemas = {}
    pdvs_path = None
    for f in raw_files:
        pfile = pq.ParquetFile(f)
        cols = lower_cols(pfile)
        schemas[f.name] = cols
        if MUST_HAVE.issubset(cols) and len(FORBIDDEN & cols) == 0:
            pdvs_path = f
            break

    if pdvs_path is None:
        details = "\n".join([f"- {k}: {sorted(list(v))}" for k, v in schemas.items()])
        raise RuntimeError("Arquivo de PDVs não identificado automaticamente. Schemas detectados:\n" + details)

    print("Arquivo de PDVs identificado:", pdvs_path.name)

    # Selecionar colunas úteis (ID e auxiliares, se existirem)
    useful = ["pdv", "categoria_pdv", "premise", "zipcode"]
    pfile = pq.ParquetFile(pdvs_path)
    cols_available = [c for c in useful if c in {f.name.lower() for f in pfile.schema_arrow}]
    read_cols = cols_available or ["pdv"]
    print("Colunas selecionadas para leitura:", read_cols)

    # Leitura por row groups
    dfs = []
    print("Row groups:", pfile.num_row_groups)
    for rg in range(pfile.num_row_groups):
        tbl = pfile.read_row_group(rg, columns=read_cols)
        dfs.append(tbl.to_pandas())

    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=read_cols)

    # Normalização
    df["pdv"] = df["pdv"].astype("string").str.strip()
    for c in ("categoria_pdv", "premise", "zipcode"):
        if c in df.columns:
            df[c] = df[c].astype("string")

    df = df.drop_duplicates(subset=["pdv"]).reset_index(drop=True)

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
