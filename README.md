# Hackathon Forecast Big Data 2025

Pipeline reprodutível para previsão semanal por PDV/SKU (jan/2023) a partir do histórico de 2022.

## Requisitos
- Python 3.10+ (Colab recomendado)
- Dependências em `requirements.txt`

## Execução (fim-a-fim)
```bash
python src/prepare_data.py
python src/make_splits.py
python src/train_baselines.py
python src/train_prophet_topn.py --top_n 200
python src/forecast_ensemble.py --top_n 200
python src/make_submission.py
```

## Estrutura
```
hackathon-forecast-2025/
├─ data/
│  ├─ raw/
│  └─ processed/
├─ notebooks/
├─ src/
├─ models/
└─ reports/
```

_Atualizado: 2025-09-13 19:20 UTC_

**Entregável final**: `reports/submission_final_JAN2023.csv` (UTF-8, `;`).
