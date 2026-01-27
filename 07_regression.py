import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path

# ==================================================
# 1. 경로 설정
# ==================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "mnt" / "data"

# ==================================================
# 2. 데이터 로드
# ==================================================
df = pd.read_csv(DATA_DIR / "public_survey_2022_2024_merged.csv")

# ==================================================
# 3. 지수 생성
# ==================================================
index_df = pd.DataFrame({
    "performance_mgmt_index": df[[f"q26_{i}" for i in range(1, 6)]].mean(axis=1),
    "job_stress_index": df[[f"q31_{i}" for i in range(1, 8)]].mean(axis=1),
    "org_commitment_index": df[[f"q35_{i}" for i in range(1, 5)]].mean(axis=1),
    "job_satisfaction_index": df[[f"q36_{i}" for i in range(1, 4)]].mean(axis=1),
})

df = pd.concat([df, index_df], axis=1)

# ==================================================
# 4. 분석용 변수 선택
# ==================================================
df_reg = df[
    [
        "performance_mgmt_index",
        "job_stress_index",
        "org_commitment_index",
        "job_satisfaction_index",
        "DM1", "DM2", "DM3", "DM4", "DM7",
        "orgtype",
        "year",
    ]
].copy()

# ==================================================
# 5. 더미 변수 생성
#   - 기준범주 자동 제외 (drop_first=True)
#   - year: 2022가 기준
# ==================================================
df_reg = pd.get_dummies(
    df_reg,
    columns=["DM1", "DM2", "DM3", "DM4", "DM7", "orgtype", "year"],
    drop_first=True
)

# ==================================================
# 6. 숫자형 강제 변환
# ==================================================
df_reg = df_reg.apply(pd.to_numeric, errors="coerce")

# ==================================================
# 7. 전부 NaN 컬럼 제거
# ==================================================
df_reg = df_reg.dropna(axis=1, how="all")

# ==================================================
# 8. 회귀 함수
# ==================================================
def run_reg(y, x_list, data):
    sub = data[[y] + x_list].dropna()

    Y = sub[y].astype(float).to_numpy()
    X = sub[x_list].astype(float).to_numpy()

    X = sm.add_constant(X)

    return sm.OLS(Y, X).fit(cov_type="HC3")

# ==================================================
# 9. 통제변수 목록
# ==================================================
core_vars = [
    "performance_mgmt_index",
    "job_stress_index",
    "org_commitment_index",
    "job_satisfaction_index",
]

control_vars = [c for c in df_reg.columns if c not in core_vars]

# ==================================================
# 10. 기본 회귀분석
# ==================================================
m1 = run_reg(
    "job_stress_index",
    ["performance_mgmt_index"] + control_vars,
    df_reg
)

m2 = run_reg(
    "org_commitment_index",
    ["performance_mgmt_index"] + control_vars,
    df_reg
)

m3 = run_reg(
    "org_commitment_index",
    ["performance_mgmt_index", "job_stress_index"] + control_vars,
    df_reg
)

# ==================================================
# 11. 부트스트랩 매개효과
# ==================================================
def bootstrap_mediation(
    data,
    x,
    m,
    y,
    controls,
    n_boot=5000,
    seed=42
):
    np.random.seed(seed)

    indirect_effects = []

    for _ in range(n_boot):
        samp = data.sample(len(data), replace=True)

        # ---------- a path: x → m ----------
        Y_a = samp[m].astype(float).to_numpy()
        X_a = samp[[x] + controls].astype(float).to_numpy()
        X_a = sm.add_constant(X_a)

        a = sm.OLS(Y_a, X_a).fit().params[1]

        # ---------- b path: m → y (x 포함) ----------
        Y_b = samp[y].astype(float).to_numpy()
        X_b = samp[[x, m] + controls].astype(float).to_numpy()
        X_b = sm.add_constant(X_b)

        b = sm.OLS(Y_b, X_b).fit().params[2]

        indirect_effects.append(a * b)

    indirect_effects = np.array(indirect_effects)

    return {
        "indirect_effect": indirect_effects.mean(),
        "ci_lower": np.percentile(indirect_effects, 2.5),
        "ci_upper": np.percentile(indirect_effects, 97.5),
    }

med_result = bootstrap_mediation(
    data=df_reg,
    x="performance_mgmt_index",
    m="job_stress_index",
    y="org_commitment_index",
    controls=[],       
    n_boot=2000       
)

print(med_result)
# ==================================================
# 12. 연도 상호작용 효과
# ==================================================
df_reg["pm_x_2023"] = df_reg["performance_mgmt_index"] * df_reg.get("year_2023", 0)
df_reg["pm_x_2024"] = df_reg["performance_mgmt_index"] * df_reg.get("year_2024", 0)

interaction_vars = (
    ["performance_mgmt_index", "job_stress_index", "pm_x_2023", "pm_x_2024"]
    + control_vars
)

m_year = run_reg(
    "org_commitment_index",
    interaction_vars,
    df_reg
)

# ==================================================
# 13. 대안 종속변수 (직무만족)
# ==================================================
m_js1 = run_reg(
    "job_satisfaction_index",
    ["performance_mgmt_index"] + control_vars,
    df_reg
)

m_js2 = run_reg(
    "job_satisfaction_index",
    ["performance_mgmt_index", "job_stress_index"] + control_vars,
    df_reg
)

# ==================================================
# 14. 결과 출력
# ==================================================
print("\n==============================")
print("[모델 1] 성과관리 → 직무스트레스")
print("==============================")
print(m1.summary())

print("\n==============================")
print("[모델 2] 성과관리 → 조직몰입")
print("==============================")
print(m2.summary())

print("\n==============================")
print("[모델 3] 성과관리 + 직무스트레스 → 조직몰입")
print("==============================")
print(m3.summary())

print("\n==============================")
print("[부트스트랩 매개효과]")
print("==============================")
print(med_result)

print("\n==============================")
print("[연도 상호작용 효과]")
print("==============================")
print(m_year.summary())

print("\n==============================")
print("[직무만족 대안 종속변수]")
print("==============================")
print(m_js1.summary())
print(m_js2.summary())
