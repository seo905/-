
# ml_model.py

from pathlib import Path

import joblib
import pandas as pd

from feature_extractor import extract_features


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR / "model" / "xgboost_model.pkl"
)

_MODEL = None
_FEATURE_NAMES = None
_THRESHOLD = None


def _load_model():
    """학습된 모델을 최초 1회 불러오기"""

    global _MODEL
    global _FEATURE_NAMES
    global _THRESHOLD

    if _MODEL is not None:
        return

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "모델 파일이 없습니다. "
            "먼저 train_model.py를 실행하세요."
        )

    saved_data = joblib.load(MODEL_PATH)

    if not isinstance(saved_data, dict):
        raise ValueError(
            "올바르지 않은 모델 파일입니다."
        )

    _MODEL = saved_data["model"]
    _FEATURE_NAMES = saved_data["feature_names"]

    _THRESHOLD = saved_data.get(
        "threshold",
        0.5
    )


def predict(features: dict) -> dict:
    """
    추출된 특징값을 이용해 피싱 URL 예측

    입력:
        extract_features()가 반환하는 특징값

    출력:
        prediction, probability, risk_level
    """

    _load_model()

    # 필수 특징 누락 여부 확인
    missing_features = [
        name
        for name in _FEATURE_NAMES
        if name not in features
    ]

    if missing_features:
        raise ValueError(
            f"필수 특징값이 누락되었습니다: {missing_features}"
        )

    # 학습 당시 특징 순서와 동일하게 구성
    input_df = pd.DataFrame(
        [[features[name] for name in _FEATURE_NAMES]],
        columns=_FEATURE_NAMES
    )

    # 피싱 확률 계산
    probabilities = _MODEL.predict_proba(input_df)[0]

    classes = list(_MODEL.classes_)

    if 1 not in classes:
        raise ValueError(
            "모델에 피싱 클래스(1)가 없습니다."
        )

    phishing_index = classes.index(1)

    phishing_prob = float(
        probabilities[phishing_index]
    )

    # 위험도 분류
    if phishing_prob >= 0.70:
        prediction_status = "phishing"
        risk_level = "위험"

    elif phishing_prob >= 0.35:
        prediction_status = "suspicious"
        risk_level = "주의"

    else:
        prediction_status = "safe"
        risk_level = "낮음"

    return {
        "prediction": prediction_status,
        "probability": round(phishing_prob, 4),
        "risk_level": risk_level,
    }


def predict_url(url: str) -> dict:
    """
    URL 입력부터 특징 추출 및 머신러닝 예측까지 수행
    """

    features = extract_features(url)

    result = predict(features)

    # 규칙 기반 분석 결과도 함께 제공
    result["hostname"] = features["hostname"]
    result["rule_score"] = features["risk_score"]
    result["rule_result"] = features["rule_result"]
    result["matched_keywords"] = features["matched_keywords"]
    result["matched_rules"] = features["matched_rules"]

    return result


if __name__ == "__main__":

    test_urls = [
        "https://www.google.com",
        "http://example-login.xyz",
        "https://example.com/account",
    ]

    for url in test_urls:

        try:
            result = predict_url(url)

            print("\nURL:", url)
            print("예측 결과:", result)

        except (ValueError, FileNotFoundError) as exc:
            print("\n분석 실패:", exc)
)

        except (ValueError, FileNotFoundError) as exc:
            print("분석 실패:", exc)
