
# train_model.py

from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

from xgboost import XGBClassifier

from feature_extractor import extract_features


BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "learning_dataset.xlsx"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "xgboost_model.pkl"


# feature_extractor.py에서 실제 반환하는 숫자형 특징만 사용
# risk_score, rule_prediction은 규칙 기반 결과이므로 학습에서 제외
FEATURE_NAMES = [
    "url_length",
    "hostname_length",
    "special_char_count",
    "has_ip",
    "has_https",
    "uses_http",
    "has_userinfo_at",
    "has_punycode",
    "has_suspicious_keyword",
    "keyword_count",
    "has_caution_tld",
    "subdomain_count",
    "has_2plus_subdomains",
    "hyphen_count",
    "hostname_hyphen_count",
    "has_hostname_hyphen",
    "hostname_digit_count",
    "has_hostname_4plus_digits",
    "has_url_length_75plus",
]


def load_dataset():
    """실제 학습 데이터셋 불러오기"""

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"학습 데이터셋이 없습니다: {DATASET_PATH}"
        )

    df = pd.read_excel(DATASET_PATH)

    df.columns = [
        str(col).strip().lower()
        for col in df.columns
    ]

    if "url" not in df.columns or "label" not in df.columns:
        raise ValueError(
            "learning_dataset.xlsx에는 URL과 Label 열이 필요합니다."
        )

    df = df.dropna(subset=["url", "label"]).copy()

    df["label"] = pd.to_numeric(
        df["label"],
        errors="coerce"
    )

    df = df.dropna(subset=["label"])

    if not df["label"].isin([0, 1]).all():
        raise ValueError(
            "Label은 0(정상), 1(피싱)으로 구성되어야 합니다."
        )

    df["label"] = df["label"].astype(int)

    if df["label"].nunique() != 2:
        raise ValueError(
            "정상 URL과 피싱 URL이 모두 있어야 합니다."
        )

    return df


def prepare_features(df):
    """URL에서 특징을 추출하여 학습 데이터 구성"""

    feature_rows = []
    labels = []

    for index, row in df.iterrows():

        try:
            features = extract_features(row["url"])

            feature_rows.append({
                name: features[name]
                for name in FEATURE_NAMES
            })

            labels.append(row["label"])

        except (ValueError, TypeError, KeyError) as exc:
            print(
                f"{index + 2}행 URL 분석 실패: {exc}"
            )

    X = pd.DataFrame(
        feature_rows,
        columns=FEATURE_NAMES
    )

    y = pd.Series(labels, dtype=int)

    if len(X) == 0 or y.nunique() != 2:
        raise ValueError(
            "학습에 사용할 수 있는 데이터가 부족합니다."
        )

    return X, y


def train_and_save_model():

    print("=== 학습 데이터 불러오기 ===")

    df = load_dataset()

    print(f"전체 데이터: {len(df)}개")
    print("\n라벨 분포:")
    print(df["label"].value_counts())

    print("\n=== URL 특징 추출 ===")

    X, y = prepare_features(df)

    print(f"유효 데이터: {len(X)}개")
    print(f"사용 특징: {len(FEATURE_NAMES)}개")

    # 학습 데이터와 검증 데이터 분리
    class_counts = y.value_counts()

    if class_counts.min() < 2:
        raise ValueError(
            "각 클래스에 최소 2개 이상의 데이터가 필요합니다."
        )

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 클래스 불균형 대응
    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()

    scale_pos_weight = negative_count / positive_count

    print("\n=== XGBoost 모델 학습 ===")

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # 검증 데이터 예측
    predictions = model.predict(X_valid)

    print("\n=== 모델 검증 결과 ===")

    print(
        "Accuracy:",
        round(accuracy_score(y_valid, predictions), 4)
    )

    print(
        "Precision:",
        round(
            precision_score(
                y_valid,
                predictions,
                zero_division=0
            ),
            4
        )
    )

    print(
        "Recall:",
        round(
            recall_score(
                y_valid,
                predictions,
                zero_division=0
            ),
            4
        )
    )

    print(
        "F1-score:",
        round(
            f1_score(
                y_valid,
                predictions,
                zero_division=0
            ),
            4
        )
    )

    print("\n상세 분류 결과:")
    print(
        classification_report(
            y_valid,
            predictions,
            zero_division=0
        )
    )

    # 모델 저장
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    save_data = {
        "model": model,
        "feature_names": FEATURE_NAMES,
        "threshold": 0.5,
    }

    joblib.dump(save_data, MODEL_PATH)

    print("\n=== 모델 저장 완료 ===")
    print(MODEL_PATH)


if __name__ == "__main__":
    train_and_save_model()
