import pandas as pd
from pathlib import Path

# --------------------------------------------------
# 데이터 로드
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "mnt" / "data"

df = pd.read_csv(DATA_DIR / "analysis_index_2022_2024.csv")

vars_core = [
    "performance_mgmt_index",
    "job_stress_index",
    "org_commitment_index",
    "job_satisfaction_index"
]

# --------------------------------------------------
# 1. Pearson 상관분석 (기본)
# --------------------------------------------------
corr_pearson = df[vars_core].corr(method="pearson")

print("\n[ Pearson 상관계수 ]")
print(corr_pearson.round(3))

# --------------------------------------------------
# 2. Spearman 상관분석 (강건성 체크)
# --------------------------------------------------
corr_spearman = df[vars_core].corr(method="spearman")

print("\n[ Spearman 상관계수 ]")
print(corr_spearman.round(3))

# --------------------------------------------------
# 3. 연도별 상관분석 (확인용)
# --------------------------------------------------
corr_by_year = {}

for y in sorted(df["year"].unique()):
    corr_by_year[y] = df[df["year"] == y][vars_core].corr(method="pearson")

# --------------------------------------------------
# 4. 저장
# --------------------------------------------------
corr_pearson.to_csv(
    DATA_DIR / "table_corr_pearson.csv",
    encoding="utf-8-sig"
)

corr_spearman.to_csv(
    DATA_DIR / "table_corr_spearman.csv",
    encoding="utf-8-sig"
)

for y, c in corr_by_year.items():
    c.to_csv(
        DATA_DIR / f"table_corr_pearson_{y}.csv",
        encoding="utf-8-sig"
    )

print("\n상관분석 결과 저장 완료")
