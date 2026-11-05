# 🛡️ AI 기반 피싱 URL 탐지 및 위험 분석 시스템

## 실행 순서
### ① 가상환경 및 라이브러리 설치
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt


### ② URL 특징 추출
python feature_extractor.py


### ③ XGBoost 모델 학습
python train_model.py

#### 생성된 파일
xgboost_model.pkl


### ④ 모델 예측 테스트
python ml_model.py


### ⑤ 테스트 데이터 성능 평가
python test.py


### 최종적으로 확인할 결과물
learning_features.xlsx
xgboost_model.pkl
test_results_detailed.xlsx
