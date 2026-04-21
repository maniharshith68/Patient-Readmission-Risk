# src/features/engineer.py

import os
import warnings
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

from src.utils.logger import get_logger

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*DtypeWarning.*")

logger = get_logger("engineer")

# ── ICD-9 diagnosis bucketing ─────────────────────────────────────────────────
def _icd9_to_category(code) -> str:
    if pd.isnull(code) or str(code).strip() in ("?", ""):
        return "Other"
    code = str(code).strip()
    if code.startswith("E") or code.startswith("V"):
        return "External/Supplemental"
    try:
        num = float(code)
    except ValueError:
        return "Other"
    if 390 <= num <= 459 or num == 785:
        return "Circulatory"
    if 460 <= num <= 519 or num == 786:
        return "Respiratory"
    if 520 <= num <= 579 or num == 787:
        return "Digestive"
    if 250 <= num <= 250.99:
        return "Diabetes"
    if 800 <= num <= 999:
        return "Injury"
    if 710 <= num <= 739:
        return "Musculoskeletal"
    if 580 <= num <= 629 or num == 788:
        return "Genitourinary"
    if 140 <= num <= 239:
        return "Neoplasms"
    return "Other"


# ── Age bracket → ordinal integer ─────────────────────────────────────────────
AGE_MAP = {
    "[0-10)": 0, "[10-20)": 1, "[20-30)": 2, "[30-40)": 3,
    "[40-50)": 4, "[50-60)": 5, "[60-70)": 6,
    "[70-80)": 7, "[80-90)": 8, "[90-100)": 9
}

# ── Medication columns ────────────────────────────────────────────────────────
MED_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "acetohexamide", "glipizide", "glyburide",
    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
    "miglitol", "troglitazone", "tolazamide", "examide",
    "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone"
]

# ── High-missingness columns to drop (>40% missing as "?") ───────────────────
HIGH_MISS_COLS = ["weight", "payer_code", "medical_specialty"]

# ── Columns to drop (identifiers / leakage) ───────────────────────────────────
DROP_COLS = [
    "encounter_id", "patient_nbr",
    "examide", "citoglipton"          # near-zero variance
]


def clean_and_encode(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Starting feature engineering pipeline...")
    df = df.copy()

    # 1 ── Replace "?" with NaN then drop high-missingness cols
    df.replace("?", np.nan, inplace=True)
    cols_to_drop = [c for c in HIGH_MISS_COLS + DROP_COLS if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)
    logger.info(f"Dropped {len(cols_to_drop)} columns: {cols_to_drop}")

    # 2 ── Recode target: 1 = readmitted <30 days, 0 = everything else
    df["readmitted"] = (df["readmitted"] == "<30").astype(int)
    pos = df["readmitted"].sum()
    logger.info(f"Target recoded — positive (<30d): {pos:,} | negative: {len(df)-pos:,} "
                f"| imbalance ratio: 1:{(len(df)-pos)//pos}")

    # 3 ── Remove rows with Unknown/Invalid gender
    df = df[~df["gender"].isin(["Unknown/Invalid"])]
    logger.info(f"Removed Unknown/Invalid gender rows — {len(df):,} rows remain")

    # 4 ── ICD-9 diagnosis bucketing
    for col in ["diag_1", "diag_2", "diag_3"]:
        if col in df.columns:
            df[col] = df[col].apply(_icd9_to_category)
    logger.info("ICD-9 codes bucketed into 9 clinical categories")

    # 5 ── Age ordinal encoding
    if "age" in df.columns:
        df["age"] = df["age"].map(AGE_MAP).fillna(4).astype(int)
    logger.info("Age brackets encoded as ordinal integers")

    # 6 ── Medication change features
    med_present = [c for c in MED_COLS if c in df.columns]

    df["num_meds_changed"] = df[med_present].apply(
        lambda row: (row.isin(["Up", "Down"])).sum(), axis=1
    )
    df["num_meds_active"] = df[med_present].apply(
        lambda row: (row != "No").sum(), axis=1
    )
    logger.info(f"Engineered medication features: num_meds_changed, num_meds_active "
                f"(from {len(med_present)} med columns)")

    # 7 ── Encode medication columns as ordinal
    med_order = {"No": 0, "Steady": 1, "Up": 2, "Down": 3}
    for col in med_present:
        if col in df.columns:
            df[col] = df[col].map(med_order).fillna(0).astype(int)

    # 8 ── Label encode remaining categoricals
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c != "readmitted"]
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])
    logger.info(f"Label encoded {len(cat_cols)} categorical columns: {cat_cols}")

    # 9 ── Fill any remaining NaNs with median
    null_counts = df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0].index.tolist()
    if cols_with_nulls:
        for col in cols_with_nulls:
            df[col].fillna(df[col].median(), inplace=True)
        logger.info(f"Filled NaNs with median in: {cols_with_nulls}")

    logger.info(f"Clean dataframe shape: {df.shape} | "
                f"Features: {df.shape[1]-1} | Rows: {df.shape[0]:,}")
    return df


def split_and_smote(df: pd.DataFrame, cfg: dict) -> tuple:
    target   = cfg["data"]["target_column"]
    test_sz  = cfg["data"]["test_size"]
    seed     = cfg["data"]["random_state"]
    k_nbrs   = cfg["smote"]["k_neighbors"]

    X = df.drop(columns=[target])
    y = df[target]

    logger.info(f"Splitting data — test_size={test_sz}, random_state={seed}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_sz, random_state=seed, stratify=y
    )
    logger.info(f"Train: {X_train.shape} | Test: {X_test.shape}")
    logger.info(f"Train class distribution before SMOTE — "
                f"0: {(y_train==0).sum():,} | 1: {(y_train==1).sum():,}")

    # SMOTE on training set ONLY
    smote = SMOTE(
        sampling_strategy=cfg["smote"]["sampling_strategy"],
        k_neighbors=k_nbrs,
        random_state=seed
    )
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    logger.info(f"After SMOTE — Train: {X_train_bal.shape} | "
                f"0: {(y_train_bal==0).sum():,} | 1: {(y_train_bal==1).sum():,}")
    logger.info(f"Test class distribution (unchanged) — "
                f"0: {(y_test==0).sum():,} | 1: {(y_test==1).sum():,}")

    return X_train_bal, X_test, y_train_bal, y_test


def save_splits(X_train, X_test, y_train, y_test, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    X_train.to_csv(os.path.join(out_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(out_dir, "X_test.csv"),  index=False)
    pd.Series(y_train, name="readmitted").to_csv(
        os.path.join(out_dir, "y_train.csv"), index=False)
    pd.Series(y_test,  name="readmitted").to_csv(
        os.path.join(out_dir, "y_test.csv"),  index=False)
    logger.info(f"Saved processed splits to {out_dir}")
    logger.info(f"  X_train: {X_train.shape} | X_test: {X_test.shape}")


def run_feature_engineering(cfg: dict) -> tuple:
    from src.ingestion.load_data import load_full_dataframe
    df  = load_full_dataframe(cfg["paths"]["raw_data"])
    df  = clean_and_encode(df)
    X_train, X_test, y_train, y_test = split_and_smote(df, cfg)
    save_splits(X_train, X_test, y_train, y_test, cfg["paths"]["processed_data"])
    logger.info("Phase 3 complete — feature engineering and SMOTE done.")
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)
    run_feature_engineering(cfg)
    print("\n✅ Phase 3 complete. Check data/processed/ and logs/engineer.log")
