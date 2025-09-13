# src/process_produtos.py
from pathlib import Path
import sys
import pyarrow.parquet as pq
import pandas as pd

def main():
    base_dir = Path(__file__).resolve().parents[1]
    data_raw  = base_dir / "data" / "raw"
    data_proc = base_dir / "data" / "processed"
    out_path  = data_proc / "produtos.parquet"

    print("BASE_DIR:", base_dir)
    print("RAW_DIR :", data_raw)
    print("PROC_DIR:", data_proc)

    raw_files = sorted(data_raw.glob("*.parquet"))
    print(f"Arquivos em data/raw (*.parquet): {len(raw_files)}")
    for f in raw_files:
        print(" -", f.name)

    if not raw_files:
        raise FileNotFoundError(f"Nenhum .parquet encontrado em {data_raw}")

    CAND_ID   = {"produto", "product_id", "sku", "id_produto", "id_sku"}
    FORBIDDEN = {"data", "quantidade", "qtd", "qtd_vendida"}

    def lower_cols(pfile: pq.ParquetFile) -> set:
        return {f.name.lower() for f in pfile.schema_arrow}

    def is_products(cols: set) -> bool:
        return (len(CAND_ID & cols) > 0) and (len(FORBIDDEN & cols) == 0)

    schemas = {}
    products_path = None
    for f in raw_files:
        pfile = pq.ParquetFile(f)
        cols = lower_cols(pfile)
        schemas[f.name] = cols
        print(f"[schema] {f.name}: {sorted(cols)}")
        if is_products(cols) and products_path is None:
            products_path = f

    if products_path is None:
        details = "\n".join([f"- {k}: {sorted(list(v))}" for k, v in schemas.items()])
        raise RuntimeError("Arquivo de produtos não identificado automaticamente. Schemas detectados:\n" + details)

    print("Arquivo de produtos identificado:", products_path.name)

    possible_id = ["produto", "product_id", "sku", "id_produto", "id_sku"]
    cols_set = schemas[products_path.name]
    id_col = next((c for c in possible_id if c in cols_set), None)
    if id_col is None:
        raise RuntimeError("Nenhuma coluna de ID de produto encontrada.")

    possible_cat = ["categoria","categoria_produto","categoria_produto_1","departamento",
                    "categoria1","subcategoria","subcategory","category","department"]
    cat_col = next((c for c in possible_cat if c in cols_set), None)

    read_cols = [id_col] + ([cat_col] if cat_col else [])
    print("Colunas selecionadas para leitura:", read_cols)

    pfile = pq.ParquetFile(products_path)
    dfs = []
    print("Row groups:", pfile.num_row_groups)
    for rg in range(pfile.num_row_groups):
        tbl = pfile.read_row_group(rg, columns=read_cols)
        dfs.append(tbl.to_pandas())

    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=read_cols)

    if id_col != "produto":
        df = df.rename(columns={id_col: "produto"})
    df["produto"] = df["produto"].astype("string").str.strip()

    if cat_col:
        if cat_col != "categoria":
            df = df.rename(columns={cat_col: "categoria"})
        df["categoria"] = df["categoria"].astype("string")

    df = df.drop_duplicates(subset=["produto"]).reset_index(drop=True)

    data_proc.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

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
