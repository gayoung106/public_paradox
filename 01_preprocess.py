import pandas as pd
from pathlib import Path

# --------------------------------------------------
# 1. 경로 설정
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "mnt" / "data"

merged_path = DATA_DIR / "public_survey_2022_2024_merged.csv"

# --------------------------------------------------
# 2. 데이터 로드
# --------------------------------------------------
df = pd.read_csv(merged_path)

print("데이터 크기:", df.shape)


# --------------------------------------------------
# 핵심 문항 번호 정의
# --------------------------------------------------
core_questions = {
    "performance_mgmt": 26,   # 성과관리
    "job_stress": 31,         # 직무스트레스
    "org_commitment": 35,     # 조직몰입
    "job_satisfaction": 36,   # 직무만족
}

selected_cols = ["year"]

for col in df.columns:
    for q_num in core_questions.values():
        if col.startswith(f"q{q_num}_"):
            selected_cols.append(col)
            break

selected_cols = sorted(set(selected_cols))

print("\n선택된 핵심 변수 목록:")
for c in selected_cols:
    print(c)

print("\n선택된 변수 개수:", len(selected_cols))


analysis_df = df[selected_cols].copy()

print("\n분석용 데이터 크기:", analysis_df.shape)

output_path = DATA_DIR / "analysis_core_2022_2024.csv"
analysis_df.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"분석용 CSV 저장 완료 → {output_path}")

