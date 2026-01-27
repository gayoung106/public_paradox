import pandas as pd
from pathlib import Path
import numpy as np

# --------------------------------------------------
# Cronbach's alpha 함수
# --------------------------------------------------
def cronbach_alpha(df_items):
    df_items = df_items.dropna()
    k = df_items.shape[1]
    variances = df_items.var(axis=0, ddof=1)
    total_var = df_items.sum(axis=1).var(ddof=1)
    alpha = (k / (k - 1)) * (1 - variances.sum() / total_var)
    return alpha

# --------------------------------------------------
# 데이터 로드
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "mnt" / "data"

df = pd.read_csv(DATA_DIR / "analysis_core_2022_2024.csv")

# --------------------------------------------------
# 변수 그룹
# --------------------------------------------------
groups = {
    "성과관리": [f"q26_{i}" for i in range(1, 6)],
    "직무스트레스": [f"q31_{i}" for i in range(1, 8)],
    "조직몰입": [f"q35_{i}" for i in range(1, 5)],
    "직무만족": [f"q36_{i}" for i in range(1, 4)],
}

# --------------------------------------------------
# 신뢰도 계산
# --------------------------------------------------
results = []

for name, cols in groups.items():
    alpha = cronbach_alpha(df[cols])
    results.append({
        "construct": name,
        "n_items": len(cols),
        "cronbach_alpha": round(alpha, 3)
    })

reliability_df = pd.DataFrame(results)

print("\nCronbach's alpha 결과")
print(reliability_df)

# --------------------------------------------------
# 저장
# --------------------------------------------------
reliability_df.to_csv(
    DATA_DIR / "table_reliability.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n신뢰도 분석 결과 저장 완료")
