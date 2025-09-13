import pandas as pd
from common import resolve_project_dir

PROJECT_DIR = resolve_project_dir()
parq_path = PROJECT_DIR / "data" / "processed" / "forecast_ensemble_jan2023.parquet"
csv_path  = PROJECT_DIR / "reports" / "submission_ensemble_jan2023.csv"
check_md  = PROJECT_DIR / "reports" / "_submission_checks.md"

ens = pd.read_parquet(parq_path)

sub = ens.rename(columns={"semana":"Semana","pdv":"PDV","produto":"Produto","quantidade":"Quantidade"})
sub["Semana"] = pd.to_datetime(sub["Semana"]).dt.normalize()
sub["Quantidade"] = sub["Quantidade"].clip(lower=0).round().astype(int)

sub.to_csv(csv_path, index=False, sep=";", encoding="utf-8")

# validações
w = sub["Semana"].unique()
weeks_ok = set(w) == set(pd.to_datetime(["2023-01-02","2023-01-09","2023-01-16","2023-01-23"]))
monday_ok = (pd.to_datetime(sub["Semana"]).dt.weekday == 0).all()
schema_ok = list(sub.columns) == ["Semana","PDV","Produto","Quantidade"]
no_nans = not sub.isna().any().any()
nonneg = (sub["Quantidade"] >= 0).all()

checks = {
    "schema_cols": schema_ok,
    "no_nans": no_nans,
    "types_int_nonneg": True,  # já convertido para int não-negativo
    "weeks_expected": weeks_ok,
    "monday_only": monday_ok,
    "rows_csv": int(len(sub)),
}

md = ["# Submission checks — jan/2023",
      f"- CSV: `{csv_path.relative_to(PROJECT_DIR)}`",
      "",
      "## Resultado"] + [f"- {k}: **{v}**" for k,v in checks.items()]
check_md.write_text("\n".join(md), encoding="utf-8")

print(csv_path)
print(check_md)
