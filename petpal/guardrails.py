"""설계서 3.3 Guardrails — 규칙 기반 1차 필터와 출력 검수 규칙.

설계 원칙: 저비용·저지연인 규칙 기반 필터를 먼저 적용하고,
통과된 것만 비용이 큰 모델 기반 판별기로 넘긴다.
"""

from __future__ import annotations

import re

# 1차 규칙 필터 — 여기서 걸리면 모델 호출 없이 즉시 차단한다.
ABUSE_PATTERNS = [
    r"몰래.*(팔|판매|거래)", r"유기동물.*(팔|되팔)", r"강아지.*공장", r"불법.*(번식|교배|분양)",
    r"(때려|학대|굶겨|버려).*(도\s*되|해도|방법)", r"안락사.*(시키는|방법)",
]
INJECTION_PATTERNS = [
    r"(이전|위).{0,6}(지시|명령|프롬프트).{0,10}(무시|잊)", r"시스템\s*프롬프트.{0,10}(출력|알려|보여)",
    r"ignore\s+(all\s+)?previous", r"system\s+prompt", r"you\s+are\s+now",
]
# 온토픽 키워드 — 하나도 없으면 오프토픽 후보로 보고 모델 판별로 넘긴다.
ON_TOPIC = [
    "유기", "구조", "입양", "보호소", "공고", "반려", "강아지", "고양이", "댕댕", "냥",
    "동반", "여행", "숙소", "펜션", "카페", "식당", "산책", "품종", "견종", "분양",
]

BLOCK_MESSAGE = {
    "abuse_request": "동물 학대나 불법 거래와 관련된 요청은 도와드릴 수 없습니다. 정식 입양 절차나 동반여행 정보는 언제든 안내해 드릴게요.",
    "injection": "시스템 설정에 대한 요청은 처리하지 않습니다. 유기동물 입양이나 반려동물 동반여행에 대해 물어봐 주세요.",
    "off_topic": "저는 유기동물 입양 상담과 반려동물 동반여행 추천을 도와드리는 '함께갈개'입니다. 관련된 질문을 해주시면 도와드릴게요.",
}

# 개인 휴대폰만 마스킹한다. 보호소 대표번호(지역번호)는 안내에 필요하므로 남긴다.
#
# 주의: \b(word boundary)는 유니코드 문자(한글 등)와 숫자가 모두 \w 로 취급되어
# "01012345678입니다" 같은 케이스에서 경계로 인식되지 않아 마스킹이 누락될 수 있다.
# 숫자/이메일 문자 셋을 기준으로 lookaround 를 사용해 누락을 줄인다.
MOBILE_RE = re.compile(r"(?<!\d)01[016-9][-.\s]?\d{3,4}[-.\s]?\d{4}(?!\d)")
EMAIL_RE = re.compile(
    r"(?<![A-Za-z0-9_.+-])[A-Za-z0-9_.+-]+@[A-Za-z0-9-]+\.[A-Za-z0-9-.]+(?![A-Za-z0-9_.+-])"
)


def rule_screen(text: str) -> tuple[str, bool]:
    """(라벨, 확정 여부). 확정이 False 면 모델 판별기로 넘긴다."""
    lowered = (text or "").lower()
    for pat in ABUSE_PATTERNS:
        if re.search(pat, text):
            return "abuse_request", True
    for pat in INJECTION_PATTERNS:
        if re.search(pat, lowered):
            return "injection", True
    if any(word in text for word in ON_TOPIC):
        return "normal", True
    return "off_topic", False  # 애매하므로 모델에게 확인시킨다


def mask_pii(text: str) -> str:
    text = MOBILE_RE.sub("010-****-****", text or "")
    return EMAIL_RE.sub("***@***", text)
