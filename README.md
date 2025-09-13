# Hackathon Forecast 2025 — Previsão de Vendas (Jan/2023)

Solução desenvolvida para o Hackathon Forecast 2025. O pipeline gera o arquivo final de previsão no **formato CSV (UTF-8)**, com **separador `;`**, no esquema exigido pelo regulamento.

---

## Entregável final

- **Arquivo:** `reports/submission_final_JAN2023.csv`  
- **Formato:** CSV (UTF-8), separador `;`  
- **Schema (colunas):**
  ```
  semana;pdv;produto;quantidade
  ```
  - **semana** (inteiro): valores **1–4** correspondendo às semanas de **02, 09, 16 e 23/01/2023**  
  - **pdv** (inteiro): código do ponto de venda  
  - **produto** (inteiro): código do SKU  
  - **quantidade** (inteiro): previsão de vendas (não negativa)

---

## Requisitos

- Python **3.10+**
- Dependências principais:
  - `pandas`, `numpy`, `pyarrow`
  - `prophet`, `cmdstanpy`, `tqdm`
- Instalação:
  ```bash
  pip install -r requirements.txt
  ```

> Observação: a reprodução **completa** (incluindo treino do Prophet) é mais demorada. Para conferência rápida do entregável, utilizar a **Rota A (rápida)** descrita abaixo.

---

## Como reproduzir

### Rota A — Geração rápida (recomendada)
Utiliza o ensemble já processado e gera o CSV final no padrão do regulamento.

1. Abrir e executar o notebook: `notebooks/00_end_to_end.ipynb`  
2. Executar a seção **Rota A — Geração rápida**  
3. Ao final, será produzido `reports/submission_final_JAN2023.csv` (UTF-8, `;`)

### Rota B — Reprocessamento completo (opcional)
Refaz todo o pipeline: treino/validação Prophet (val4 + produção Jan/2023), ensemble e exportação.

1. No notebook `notebooks/00_end_to_end.ipynb`, executar a seção **Rota B — Reprocessamento completo**  
2. Ao final, reutilizar a seção de exportação para gerar `reports/submission_final_JAN2023.csv`

---

## Execução por linha de comando (alternativa ao notebook)

```bash
# 1) Instalar dependências
pip install -r requirements.txt

# 2) (Opcional) Treino/validação Prophet (val4) e modo produção Jan/2023
python -u src/train_prophet_topn.py \
  --top_n 200 \
  --changepoint_prior_scale 0.8 \
  --val_split val4 \
  --predict_jan2023 \
  --out_parquet data/processed/prophet_topN_jan2023_preds.parquet

# 3) Ensemble (gera data/processed/forecast_ensemble_jan2023.parquet)
python -u src/forecast_ensemble.py \
  --out_parquet data/processed/forecast_ensemble_jan2023.parquet

# 4) Exportação final (UTF-8, ';')
python - <<'PY'
from pathlib import Path
import pandas as pd, numpy as np

root = Path(".")
ens_pq = root/"data/processed/forecast_ensemble_jan2023.parquet"
out_csv = root/"reports/submission_final_JAN2023.csv"

df = pd.read_parquet(ens_pq)

# Normalização de nomes
ren = {}
for c in df.columns:
    lc = c.lower()
    if lc == "sku": ren[c] = "produto"
    elif lc == "pdv": ren[c] = "pdv"
    elif lc in ("sku_id","produto_id","product","product_id"): ren[c] = "produto"
    elif lc in ("pdv_id","store","store_id"): ren[c] = "pdv"
    elif lc in ("date","dt","ds","semana"): ren[c] = "ds"
    elif lc in ("pred","prediction","forecast","y_pred","yhat_prophet","yhat_baseline","ens_pred","yhat_ensemble","yhat"):
        ren[c] = "yhat"
df = df.rename(columns=ren)

# Extrai chaves de "id" no formato "pdv|sku" (se necessário)
if "id" in df.columns and (("pdv" not in df.columns) or ("produto" not in df.columns)):
    ids = df["id"].astype(str).str.split("|", n=1, expand=True)
    if ids.shape[1] == 2:
        df["pdv"] = ids[0]
        df["produto"] = ids[1]

# Tipos
for k in ("pdv","produto"):
    if k in df.columns: df[k] = df[k].astype(str)

# Datas alvo (segundas-feiras do mês)
df["ds"] = pd.to_datetime(df.get("ds"), errors="coerce")
try: df["ds"] = df["ds"].dt.tz_localize(None)
except Exception: pass
ref = [pd.Timestamp(2023,1,d) for d in (2,9,16,23)]
df = df[df["ds"].isin(ref)].copy()

# Semana 1..4
map_sem = {pd.Timestamp(2023,1,2):1, pd.Timestamp(2023,1,9):2,
           pd.Timestamp(2023,1,16):3, pd.Timestamp(2023,1,23):4}
df["semana"] = df["ds"].map(map_sem).astype(int)

# Coluna de previsão (usa 'ma4' se existir; senão, 'yhat')
pred_col = "ma4" if "ma4" in df.columns else "yhat"
if pred_col not in df.columns:
    raise ValueError("Nenhuma coluna de previsão encontrada (yhat/ma4).")

out = df[["semana","pdv","produto",pred_col]].copy()
out = out.rename(columns={pred_col:"quantidade"})
out["quantidade"] = np.clip(out["quantidade"].round().astype(int), 0, None)

out.to_csv(out_csv, sep=";", index=False, encoding="utf-8")
print("OK - salvo:", out_csv, "| linhas:", len(out))
PY
```

---

## Estrutura do repositório

```
.
├── data/
│   ├── raw/                          # dados brutos (não versionados)
│   └── processed/
│       ├── baseline_preds.parquet
│       └── forecast_ensemble_jan2023.parquet
├── notebooks/
│   ├── 00_end_to_end.ipynb           # pipeline reprodutível (curto)
│   └── archive/
│       └── colab_worklog.ipynb       # trabalho exploratório (opcional)
├── reports/
│   ├── submission_final_JAN2023.csv  # arquivo final (UTF-8, ';')
│   └── _submission_checks.md         # checagens auxiliares
├── src/
│   ├── train_prophet_topn.py
│   └── forecast_ensemble.py
├── requirements.txt
└── README.md
```

---

## Notas de reprodução

- O arquivo `reports/submission_final_JAN2023.csv` é produzido a partir do ensemble
  `data/processed/forecast_ensemble_jan2023.parquet`.  
- A etapa de treino Prophet (val4 + produção Jan/2023) pode ser reexecutada via **Rota B**,
  gerando novamente o ensemble antes da exportação final.

---

## Licença

Distribuído sob a licença MIT (ou equivalente, conforme aplicável).
