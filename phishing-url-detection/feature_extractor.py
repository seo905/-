"""URL 특징 추출 + 확정된 20점 규칙 점수 계산.

설치: pip install pandas openpyxl tldextract
실행: python feature_extractor.py
입력: learning_dataset.xlsx (URL 열 필수)
출력: learning_features.xlsx
주의: URL에 접속하지 않으며, 이 점수는 초기 규칙 기반 분류 결과입니다.
"""
import ipaddress
import re
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd
import tldextract

INPUT_FILE = Path(__file__).resolve().parent / "learning_dataset.xlsx"
OUTPUT_FILE = Path(__file__).resolve().parent / "learning_features.xlsx"
THRESHOLD = 2
KEYWORDS = (
    "login", "signin", "verify", "secure", "account", "update",
    "banking", "password", "payment", "support", "authentication",
)
# '등'으로 언급된 추가 TLD는 확정되지 않았으므로 명시된 다섯 개만 적용
CAUTION_TLDS = {"shop", "cfd", "sbs", "xyz", "top"}
SPECIAL_CHARS = "@_%?&=+#"
EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=())

# 규칙명, 결과 특징명, 가중치: 합계 20점
RULES = (
    ("IP 주소 직접 사용", "has_ip", 4),
    ("사용자 정보 영역 @", "has_userinfo_at", 3),
    ("Punycode", "has_punycode", 2),
    ("의심 키워드", "has_suspicious_keyword", 2),
    ("주의 대상 TLD", "has_caution_tld", 2),
    ("호스트 하이픈", "has_hostname_hyphen", 2),
    ("호스트 숫자 4개 이상", "has_hostname_4plus_digits", 2),
    ("서브도메인 2단계 이상", "has_2plus_subdomains", 1),
    ("URL 길이 75자 이상", "has_url_length_75plus", 1),
    ("HTTP 사용", "uses_http", 1),
)
assert sum(weight for _, _, weight in RULES) == 20


def extract_features(raw_url):
    """문자열만 분석합니다. 외부 접속은 하지 않습니다."""
    if pd.isna(raw_url) or not str(raw_url).strip():
        raise ValueError("URL이 비어 있습니다.")
    url = str(raw_url).strip()
    has_scheme = bool(re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", url))
    parsed = urlsplit(url if has_scheme else "//" + url)
    try:
        hostname = (parsed.hostname or "").lower().rstrip(".")
        port = parsed.port  # 잘못된 포트 값 검증
    except ValueError as exc:
        raise ValueError(f"호스트 또는 포트 형식 오류: {exc}") from exc
    if not hostname:
        raise ValueError("호스트를 찾을 수 없습니다.")

    try:
        ipaddress.ip_address(hostname)
        has_ip = 1
    except ValueError:
        has_ip = 0

    if has_ip:
        registered_domain, domain_suffix, subdomain_count = hostname, "", 0
    else:
        parts = EXTRACTOR(hostname)
        registered_domain = ".".join(p for p in (parts.domain, parts.suffix) if p)
        domain_suffix = parts.suffix
        subdomain_count = len([p for p in parts.subdomain.split(".") if p])

    matched = [
        word for word in KEYWORDS
        if re.search(r"(?<![a-z])" + re.escape(word) + r"(?![a-z])", url.lower())
    ]
    hostname_digit_count = sum(char.isascii() and char.isdigit() for char in hostname)
    # '@'는 경로나 쿼리가 아니라 사용자 정보 영역에 있는 경우만 해당
    has_userinfo_at = int("@" in parsed.netloc)
    has_https = int(parsed.scheme.lower() == "https")
    uses_http = int(parsed.scheme.lower() == "http")
    features = {
        "url_length": len(url),
        "hostname": hostname,
        "registered_domain": registered_domain,
        "domain_suffix": domain_suffix,
        "hostname_length": len(hostname),
        "special_char_count": sum(url.count(char) for char in SPECIAL_CHARS),
        "has_ip": has_ip,
        "has_https": has_https,
        "uses_http": uses_http,
        "has_userinfo_at": has_userinfo_at,
        "has_punycode": int(any(label.startswith("xn--") for label in hostname.split("."))),
        "has_suspicious_keyword": int(bool(matched)),
        "keyword_count": len(matched),
        "matched_keywords": ", ".join(matched),
        "has_caution_tld": int(domain_suffix.lower() in CAUTION_TLDS),
        "subdomain_count": subdomain_count,
        "has_2plus_subdomains": int(subdomain_count >= 2),
        "hyphen_count": url.count("-"),
        "hostname_hyphen_count": hostname.count("-"),
        "has_hostname_hyphen": int("-" in hostname),
        "hostname_digit_count": hostname_digit_count,
        "has_hostname_4plus_digits": int(hostname_digit_count >= 4),
        "has_url_length_75plus": int(len(url) >= 75),
    }
    score = sum(features[column] * weight for _, column, weight in RULES)
    features["risk_score"] = score
    features["rule_prediction"] = int(score >= THRESHOLD)  # 1=피싱, 0=정상
    features["rule_result"] = "피싱" if score >= THRESHOLD else "정상"
    features["matched_rules"] = ", ".join(name for name, column, _ in RULES if features[column])
    return features


def main():
    data = pd.read_excel(INPUT_FILE)
    columns = {str(column).strip().lower(): column for column in data.columns}
    if "url" not in columns:
        raise ValueError("엑셀에 URL 열이 필요합니다.")
    results = []
    for index, url in enumerate(data[columns["url"]], start=2):
        try:
            results.append(extract_features(url))
        except (ValueError, TypeError) as exc:
            results.append({"parse_error": str(exc)})
            print(f"엑셀 {index}행 확인 필요: {exc}")
    features = pd.DataFrame(results)
    duplicate_columns = set(data.columns) & set(features.columns)
    if duplicate_columns:
        raise ValueError(f"기존 열과 추출 특징 열이 중복됩니다: {sorted(duplicate_columns)}")
    output = pd.concat([data.reset_index(drop=True), features], axis=1)
    output.to_excel(OUTPUT_FILE, index=False)
    print(f"완료: {len(output)}개 URL → {OUTPUT_FILE}")
    print(f"점수 기준: {THRESHOLD}점 이상 피싱, 1=피싱/0=정상")


if __name__ == "__main__":
    main()

