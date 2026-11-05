# 🛡️ AI 기반 피싱 URL 탐지 및 위험 분석 시스템

## 실행 순서
### ① 가상환경 및 라이브러리 설치
python -m venv venv <br>
venv\Scripts\activate<br>
python -m pip install --upgrade pip <br>
pip install -r requirements.txt <br>


### ② URL 특징 추출
python feature_extractor.py<br>


### ③ XGBoost 모델 학습
python train_model.py<br>

#### 생성된 파일
xgboost_model.pkl<br>


### ④ 모델 예측 테스트
python ml_model.py<br>


### ⑤ 테스트 데이터 성능 평가
python test.py<br>


### 최종적으로 확인할 결과물
learning_features.xlsx<br>
xgboost_model.pkl<br>
test_results_detailed.xlsx<br>
