import pandas as pd
import pyreadstat
from functools import reduce
from pathlib import Path

# --------------------------------------------------
# 0. 개요
# 2022년, 2023년, 2024년 공공데이터 설문조사 데이터(sav)를 불러와
# 공통 변수만 추출하여 하나의 데이터프레임으로 병합한 후 CSV로 저장
# --------------------------------------------------

# --------------------------------------------------
# 1. 프로젝트 기준 경로
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "mnt" / "data"

files = {
    2022: DATA_DIR / "public_survey_2022.SAV",
    2023: DATA_DIR / "public_survey_2023.SAV",
    2024: DATA_DIR / "public_survey_2024.sav",
}

# --------------------------------------------------
# 2. 연도별 데이터 로드
# --------------------------------------------------
dfs = {}

for year, path in files.items():
    if not path.exists():
        raise FileNotFoundError(f"{path} 파일이 존재하지 않습니다.")

    df, meta = pyreadstat.read_sav(str(path))
    df["year"] = year
    dfs[year] = df

    print(f"{year} loaded | rows: {df.shape[0]}, cols: {df.shape[1]}")

# --------------------------------------------------
# 3. 공통 변수 확인
# --------------------------------------------------
common_columns = reduce(
    lambda x, y: x.intersection(y),
    [set(df.columns) for df in dfs.values()]
)

common_columns = sorted(common_columns)
print(f"\n공통 변수 수: {len(common_columns)}")

# --------------------------------------------------
# 4. 공통 변수만 추출
# --------------------------------------------------
dfs_common = [df[common_columns].copy() for df in dfs.values()]

# --------------------------------------------------
# 5. 데이터 병합
# --------------------------------------------------
merged_df = pd.concat(dfs_common, axis=0, ignore_index=True)

print("\n병합 완료")
print("최종 데이터 크기:", merged_df.shape)

# --------------------------------------------------
# 6. 저장
# --------------------------------------------------
output_path = DATA_DIR / "public_survey_2022_2024_merged.csv"
merged_df.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"CSV 저장 완료 → {output_path}")
