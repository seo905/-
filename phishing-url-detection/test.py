import pandas as pd
from feature_extractor import extract_features

def check_rules(features):
    rule_score = 0
    if features.get("has_ip"): rule_score += 4
    if features.get("has_userinfo_at"): rule_score += 3
    if features.get("has_punycode"): rule_score += 2
    if features.get("has_suspicious_keyword"): rule_score += 2
    if features.get("has_caution_tld"): rule_score += 2
    if features.get("has_hostname_hyphen"): rule_score += 2
    if features.get("has_hostname_4plus_digits"): rule_score += 2
    if features.get("has_2plus_subdomains"): rule_score += 1
    if features.get("has_url_length_75plus"): rule_score += 1
    if features.get("uses_http"): rule_score += 1

    return {"rule_score": rule_score, "is_phishing": rule_score >= 2}

def evaluate_performance(test_file_path):
    df = pd.read_excel(test_file_path)
    TP, TN, FP, FN = 0, 0, 0, 0
    
    # 결과를 저장할 리스트
    results = []
    
    for index, row in df.iterrows():
        url = row['URL']
        actual_label = row['Label']
        is_actual_phishing = (actual_label == 1) 
        
        try:
            features = extract_features(url)
            rule_result = check_rules(features)
            predicted_phishing = rule_result['is_phishing']
            score = rule_result['rule_score']
        except Exception as e:
            predicted_phishing = False
            score = 0
            
        # 개별 URL의 분류 결과 판별
        if is_actual_phishing and predicted_phishing: 
            TP += 1
            result_type = 'TP (탐지 성공)'
        elif not is_actual_phishing and not predicted_phishing: 
            TN += 1
            result_type = 'TN (정상 분류)'
        elif not is_actual_phishing and predicted_phishing: 
            FP += 1
            result_type = 'FP (오탐 - 정상인데 피싱으로 분류)'
        elif is_actual_phishing and not predicted_phishing: 
            FN += 1
            result_type = 'FN (미탐 - 피싱인데 정상으로 분류)'
            
        # 결과 리스트에 데이터 추가
        results.append({
            'URL': url,
            '실제 라벨': '피싱' if is_actual_phishing else '정상',
            '예측 라벨': '피싱' if predicted_phishing else '정상',
            '위험 점수': score,
            '분류 결과': result_type
        })

    # 기존 성능 평가 지표 출력
    total = TP + TN + FP + FN
    accuracy = (TP + TN) / total if total > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    
    print("=== 테스트 데이터 성능 평가 결과표 ===")
    print(f"정확도(Accuracy): {accuracy * 100:.1f}%")
    print(f"정밀도(Precision): {precision * 100:.1f}%")
    print(f"재현율(Recall): {recall * 100:.1f}%")
    print(f"오탐 건수(False Positives): {FP}개")
    
    # 분석 결과를 엑셀 파일로 변환하여 저장
    results_df = pd.DataFrame(results)
    results_df.to_excel("test_results_detailed.xlsx", index=False)
    print("\n상세 분류 결과가 'test_results_detailed.xlsx' 파일로 저장되었습니다.")

if __name__ == "__main__":
    evaluate_performance('test_dataset.xlsx')
