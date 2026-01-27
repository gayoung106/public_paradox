import pandas as pd
from pathlib import Path

# --------------------------------------------------
# 데이터 로드
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "mnt" / "data"

df = pd.read_csv(DATA_DIR / "analysis_core_2022_2024.csv")

# --------------------------------------------------
# 지수 생성
# --------------------------------------------------
df["performance_mgmt_index"] = df[[f"q26_{i}" for i in range(1, 6)]].mean(axis=1)
df["job_stress_index"] = df[[f"q31_{i}" for i in range(1, 8)]].mean(axis=1)
df["org_commitment_index"] = df[[f"q35_{i}" for i in range(1, 5)]].mean(axis=1)
df["job_satisfaction_index"] = df[[f"q36_{i}" for i in range(1, 4)]].mean(axis=1)

# --------------------------------------------------
# 저장
# --------------------------------------------------
output_path = DATA_DIR / "analysis_index_2022_2024.csv"
df.to_csv(output_path, index=False, encoding="utf-8-sig")

print("지수 생성 완료 →", output_path)
