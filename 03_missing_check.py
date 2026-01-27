import pandas as pd
from pathlib import Path

# --------------------------------------------------
# 1. 데이터 로드
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "mnt" / "data"

df = pd.read_csv(DATA_DIR / "analysis_core_2022_2024.csv")

# --------------------------------------------------
# 2. 분석 변수 정의
# --------------------------------------------------
groups = {
    "성과관리": [f"q26_{i}" for i in range(1, 6)],
    "직무스트레스": [f"q31_{i}" for i in range(1, 8)],
    "조직몰입": [f"q35_{i}" for i in range(1, 5)],
    "직무만족": [f"q36_{i}" for i in range(1, 4)],
}

# --------------------------------------------------
# 3. 전체 표본 기술통계
# --------------------------------------------------
desc_tables = []

for group, cols in groups.items():
    stats = df[cols].describe().T
    stats["variable"] = stats.index
    stats["construct"] = group
    desc_tables.append(stats)

desc_df = pd.concat(desc_tables)

desc_df = desc_df[[
    "construct", "variable", "count", "mean", "std", "min", "max"
]]

print(desc_df)

# --------------------------------------------------
# 4. 연도별 기술통계 (중요)
# --------------------------------------------------
yearly_desc = (
    df
    .groupby("year")[sum(groups.values(), [])]
    .agg(["mean", "std"])
)

# --------------------------------------------------
# 5. 저장
# --------------------------------------------------
desc_df.to_csv(
    DATA_DIR / "table_descriptive_overall.csv",
    index=False,
    encoding="utf-8-sig"
)

yearly_desc.to_csv(
    DATA_DIR / "table_descriptive_by_year.csv",
    encoding="utf-8-sig"
)

print("기술통계표 저장 완료")
