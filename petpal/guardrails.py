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
    r"(이전|위).{0,6}(지시|명령|프롬프트).{0,10}(무시|잊)",
    r"시스템\s*프롬프트.{0,10}(출력|알려|보여)",
    r"(지시문|프롬프트|규칙).{0,6}(전부|모두|그대로).{0,6}(보여|출력|알려)",
    # 역할 재지정형 — "너는 20년차 셰프고…" 같은 페르소나 덮어쓰기
    r"너는\s*(이제|지금부터)?\s*.{0,20}?(셰프|요리사|의사|변호사|해커|개발자|교사|작가|비서|상담사)",
    r"(지금부터|이제부터)\s*너는", r"역할을?\s*(맡아|바꿔|해줘|수행)", r"\S+인\s*척\s*(해|하고)",
    r"ignore\s+(all\s+)?previous", r"system\s+prompt", r"you\s+are\s+now",
    r"act\s+as\s+", r"pretend\s+(to\s+be|you)", r"role[-\s]?play",
    r"(원래|기존)\s*(역할|설정|지시)", r"역할\s*말고", r".{0,12}처럼\s*(글|말|답|행동|대답)",
]

# 온토픽 키워드가 있어도 이런 표현이 섞이면 규칙으로 통과시키지 않고 모델에 넘긴다.
# ("여행 블로거처럼 글 써줘" 처럼 온토픽 단어를 끼워 규칙을 우회하는 것을 막는다)
ROLE_HINTS = [
    r"역할", r"프롬프트", r"지시문", r"설정을?\s*(무시|바꿔)", r"처럼\s*(글|말|답|행동|대답)",
    r"인\s*척", r"ignore", r"system", r"assistant", r"jailbreak", r"제약\s*(없|해제)",
]
# 온토픽 키워드 — 하나도 없으면 오프토픽 후보로 보고 모델 판별로 넘긴다.
ON_TOPIC = [
    "유기", "구조", "입양", "보호소", "공고", "반려", "강아지", "고양이", "댕댕", "냥",
    "동반", "여행", "숙소", "펜션", "카페", "식당", "산책", "품종", "견종", "분양",
    "동물", "펫", "보호중", "중성화", "예방접종", "수의", "사료", "목줄",
]

BLOCK_MESSAGE = {
    "abuse_request": "동물 학대나 불법 거래와 관련된 요청은 도와드릴 수 없습니다. 정식 입양 절차나 동반여행 정보는 언제든 안내해 드릴게요.",
    "injection": "시스템 설정에 대한 요청은 처리하지 않습니다. 유기동물 입양이나 반려동물 동반여행에 대해 물어봐 주세요.",
    "off_topic": "저는 유기동물 입양 상담과 반려동물 동반여행 추천만 도와드리는 '함께갈개'입니다. 다른 분야는 답변드리기 어려워요. 입양이나 동반여행에 대해 물어봐 주세요.",
}

# 개인 휴대폰만 마스킹한다. 보호소 대표번호(지역번호)는 안내에 필요하므로 남긴다.
MOBILE_RE = re.compile(r"\b01[016-9][-.\s]?\d{3,4}[-.\s]?\d{4}\b")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")


def rule_screen(text: str) -> tuple[str, bool]:
    """(라벨, 확정 여부). 확정이 False 면 모델 판별기로 넘긴다."""
    lowered = (text or "").lower()
    for pat in ABUSE_PATTERNS:
        if re.search(pat, text):
            return "abuse_request", True
    for pat in INJECTION_PATTERNS:
        if re.search(pat, lowered):
            return "injection", True
    suspicious = any(re.search(pat, lowered) for pat in ROLE_HINTS)
    if suspicious:
        # 모델이 최종 판단하되, 호출이 실패하면 injection 으로 막는다(fail-closed).
        return "injection", False
    if any(word in text for word in ON_TOPIC):
        return "normal", True
    return "off_topic", False  # 애매하므로 모델에게 확인시킨다


def mask_pii(text: str) -> str:
    text = MOBILE_RE.sub("010-****-****", text or "")
    return EMAIL_RE.sub("***@***", text)
