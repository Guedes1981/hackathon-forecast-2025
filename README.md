# Hackathon Forecast Big Data 2025

## Entregável final
- **Path:** `reports/submission_final_JAN2023.csv`
- **Formato:** CSV (UTF-8, separador `;`)
- **Esquema (inteiros):** `semana;pdv;produto;quantidade`
- **Semanas previstas:** 1..4 (02, 09, 16, 23 jan/2023)
- **Total de linhas:** 4.177.240
- **Checksums:**
  - **MD5:** `1178999f1b2856f95ec355210ee6ec63`
  - **SHA256:** `138912d3adacdce5a29cbdb348d3221cdcdd094dd35f41908674e6fe7719e8a5`

## Reprodutibilidade
- **Ambiente:** Google Colab (Linux, Python 3.12)
- **Diretório de trabalho esperado:** `/content/drive/MyDrive/hackathon-forecast-2025`

### Passo a passo
1) **Preparar ambiente**
```bash
pip install -r requirements.txt
# ou, no Colab, instale as libs usadas nos scripts (pandas, pyarrow, prophet/cmdstanpy etc.)
```

2) **Treinar Prophet Top-N (val4) e salvar métricas**
```bash
python -u src/train_prophet_topn.py   --top_n 200   --changepoint_prior_scale 0.8   --val_split val4   --out_parquet data/processed/prophet_topN_val4_preds.parquet   --report reports/_prophet_val4_metrics.csv
```

3) **(Produção) Gerar previsões de Jan/2023 com Prophet**
> O script recorta Jan/2023 a partir do resultado de validação e salva o parquet de produção.
```bash
python -u src/train_prophet_topn.py   --predict_jan2023   --out_parquet data/processed/prophet_topN_jan2023_preds.parquet
# Obs.: se não houver saídas do Prophet para Jan, o ensemble usa fallback (baseline)
# apenas para permitir a composição.
```

4) **Rodar Ensemble (Prophet + Baseline) e exportar submissão**
```bash
python -u src/forecast_ensemble.py   --prophet_parquet data/processed/prophet_topN_jan2023_preds.parquet   --baseline_parquet data/processed/baseline_preds.parquet   --out_parquet data/processed/forecast_ensemble_jan2023.parquet

# Gerar CSV final (UTF-8, ';', schema exigido)
python -u src/make_submission.py   --in_parquet data/processed/forecast_ensemble_jan2023.parquet   --out_csv reports/submission_final_JAN2023.csv   --checks_md reports/_submission_checks.md
```

5) **Verificações rápidas**
```bash
head -n 3 reports/submission_final_JAN2023.csv
# esperado:
# semana;pdv;produto;quantidade
# 1;...;...;...

# checksums:
md5sum    reports/submission_final_JAN2023.csv
sha256sum reports/submission_final_JAN2023.csv
```

## Estrutura principal
```
src/
 ├─ train_prophet_topn.py      # treina/valida Prophet; modo produção --predict_jan2023
 ├─ forecast_ensemble.py       # junta Prophet + baseline; calcula MA(4) e composição
 └─ make_submission.py         # normaliza, valida e salva CSV final no schema do regulamento
data/
 └─ processed/
     ├─ baseline_preds.parquet
     ├─ prophet_topN_val4_preds.parquet
     └─ prophet_topN_jan2023_preds.parquet
reports/
 ├─ submission_final_JAN2023.csv   # arquivo submetido
 └─ _submission_checks.md          # sanity checks (datas, contagens, nulos, etc.)
```

## Observações
- Grandes artefatos gerados não são versionados (`.gitignore`); versiona-se apenas código e instruções.
- Para reprodução integral, garanta que os **paths** de entrada/saída existam conforme os comandos.

## Contato & autoria
- **Autor:** Murilo Brito Guedes — <guedes366@hotmail.com>
- **Atualizado:** 2025-09-13 19:20 UTC
