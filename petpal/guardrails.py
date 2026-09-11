"""설계서 3.3 Guardrails — 규칙 기반 1차 필터와 출력 검수 규칙.

설계 원칙: 저비용·저지연인 규칙 기반 필터를 먼저 적용하고,
통과된 것만 비용이 큰 모델 기반 판별기로 넘긴다.
"""

from __future__ import annotations

import re
import unicodedata

# 1차 규칙 필터 — 명백한 위해·거래만 확정 차단하고 나머지는 모델이 의미를 판별한다.
# 띄어쓰기·제로폭 문자·호환문자를 이용한 우회를 줄이기 위해 입력을 먼저 정규화한다.
_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]")


def normalize_for_screening(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text or "").lower()
    normalized = _ZERO_WIDTH_RE.sub("", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


# 요청·허용·실행 의도를 나타내는 표현. 단순히 결과나 위험성을 묻는 문장과 구분한다.
_ACTION_REQUEST = (
    r"(해\s*줘|알려\s*줘|방법|요령|절차|계획|계약서|가격|팁|사용\s*법|훈련|"
    r"사용(?:하(?:는|는)?\s*)?법|"
    r"도\s*[되돼]|해도\s*[되돼]|어도\s*[되돼]|되나|돼나|될까|괜찮|무방|효과|하고\s*싶)"
)
_HARM = (
    r"(때리|때려|패(?:는|도|기|려고)|걷어\s*차|발로\s*차|밟|던지|"
    r"목(?:을)?\s*(?:조르|졸)|굶기|굶겨|굶길|학대|방치|죽이|죽여|죽일|"
    r"가둬|가두|묶어\s*[놓놔둬두]|아프게|다치게|체벌|매질|"
    r"유기해|내다\s*버리|갖다\s*버리|버려(?!진)|꼬리.{0,5}자르|귀.{0,5}자르|"
    r"전기\s*충격|투견|싸움\s*붙|(?:차|자동차).{0,8}(?:안|내부).{0,8}(?:두|놔|방치)|"
    r"독(?:약|극물)?\s*(?:먹이|주입)|"
    r"(?:초콜릿|농약|부동액).{0,8}(?:먹이|먹여|주))"
)
_TRADE = (
    r"(재\s*판매|되\s*팔|되팔이|전매|리셀|re[-\s]?sell|위탁\s*판매|경매|"
    r"상업적?\s*(?:판매|거래|분양|양도)|"
    r"영리\s*목적|수익화|판매\s*계약|가격\s*책정|구매자\s*심사)"
)
_MONEY_TRANSFER = (
    r"((?:사례비|돈|대가|수익|이익).{0,18}(?:받|벌).{0,18}(?:보내|넘기|양도|분양)|"
    r"(?:보내|넘기|양도|분양|판매|팔아?).{0,18}(?:사례비|돈|대가|수익|이익|돈벌|장사))"
)
_BREEDING_FOR_PROFIT = r"((?:번식|교배).{0,24}(?:수익|이익|돈|판매|분양|장사)|(?:수익|이익|돈벌|장사).{0,24}(?:번식|교배))"

# 위해 표현을 언급하더라도 신고·예방·치료·위험성 설명을 구하는 요청은 확정 차단하지 않는다.
_PROTECTIVE_CONTEXT = (
    r"(신고|예방|방지|막(?:기|는)|금지|반대|구조|구해|치료|응급|대처|발견|"
    r"처벌|법적|위험|안\s*되|하지\s*말|하면\s*안|이유|결과|영향|문제|피해|"
    r"어떻게\s*되|무슨\s*일|어떤\s*영향)"
)

ABUSE_PATTERNS = [
    _HARM + r".{0,24}" + _ACTION_REQUEST,
    _ACTION_REQUEST + r".{0,24}" + _HARM,
    r"몰래.{0,30}(?:팔아?|판매|거래|넘기|양도|경매)",
    r"(?:강아지|고양이|반려견|동물).{0,20}(?:공장|투견)",
    r"안락사.{0,12}(?:시키|방법|절차)",
    _TRADE,
    _MONEY_TRANSFER,
    _BREEDING_FOR_PROFIT,
    r"불법.{0,12}(?:번식|교배|분양)",
    r"판매.{0,8}(?:하려|하고\s*싶|할까|해도|하면|목적)",
    r"(?:입양|구조|유기|보호소).{0,30}(?:판매|되팔|재판매|전매|경매|팔(?:아|려고|고\s*싶|면|까))",
]

# 온토픽 키워드가 있어도 이런 표현이 섞이면 규칙으로 통과시키지 않는다.
# '강아지' 같은 안전한 단어가 문장 전체를 화이트리스트 처리하는 것을 막는다.
HARM_HINTS = [
    _HARM, _TRADE, _MONEY_TRANSFER, _BREEDING_FOR_PROFIT,
    r"혼내", r"벌을?\s*주", r"실험", r"번식", r"교배", r"팔아", r"판매", r"양도", r"넘기",
    # 방임형 — "물 안 줘도 돼?" 와 "물 안 주면 어떻게 되나요?" 는 뜻이 달라 모델이 가른다.
    r"(밥|물|사료|끼니|산책).{0,8}안\s*(줘|주|시켜)",
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

# 스코프 밖(스포츠 승패 예측/요리 레시피/금융 시세 등) 신호.
# 온토픽 단어를 섞어도 여기 걸리면 규칙으로 off_topic 확정(모델 호출 없이 차단)한다.
OFF_TOPIC_PATTERNS = [
    # 스포츠 승패/우승 예측
    r"(야구|축구|농구|배구|epl|mlb|kbo).{0,12}(우승|승리|이길|질|승패|예측|배당|스코어)",
    r"(우승|승리|이길|질|승패|예측|배당|스코어).{0,12}(야구|축구|농구|배구|epl|mlb|kbo)",
    # 요리/레시피
    r"(레시피|만드는\s*법|요리).{0,16}(스파게티|파스타|토마토)",
    r"(스파게티|파스타).{0,16}(레시피|만드는\s*법|요리)",
    r"(김치찌개|된장찌개|불고기|라면).{0,16}(레시피|끓이는\s*법|만드는\s*법|요리)",
    # 금융/시세
    r"(코스피|코스닥|주가|주식|환율|비트코인|코인|시세)",
]

TRADE_MESSAGE = (
    "입양 동물을 영리 목적으로 판매·전매하는 일은 입양계약 위반이자 동물보호법 위반 소지가 있어 "
    "도와드릴 수 없습니다.\n"
    "더 이상 돌보기 어려운 사정이라면 입양처나 관할 보호소에 먼저 연락해 재입양 절차를 상담하시기 바랍니다. "
    "정식 입양 절차나 반려동물 동반여행 정보는 언제든 안내해 드릴게요."
)
HARM_MESSAGE = (
    "동물에게 해가 되는 행동은 안내해 드릴 수 없습니다. 동물보호법상 학대·유기는 처벌 대상입니다.\n"
    "훈련이나 돌봄 때문에 힘드시다면 수의사나 반려동물 행동 전문가 상담을 권해 드립니다. "
    "입양 절차나 반려동물 동반여행 정보는 언제든 도와드릴게요."
)

BLOCK_MESSAGE = {
    "abuse_request": HARM_MESSAGE,
    "injection": "시스템 설정에 대한 요청은 처리하지 않습니다. 유기동물 입양이나 반려동물 동반여행에 대해 물어봐 주세요.",
    "off_topic": "저는 유기동물 입양 상담과 반려동물 동반여행 추천만 도와드리는 '함께갈개'입니다. 다른 분야는 답변드리기 어려워요. 입양이나 동반여행에 대해 물어봐 주세요.",
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
    lowered = normalize_for_screening(text)
    # 조사 사이를 벌리는 우회("때 려도")도 검사하되, 원문 경계가 필요한 패턴은 lowered 로 본다.
    compact = re.sub(r"[\s._·,!?/\\-]+", "", lowered)

    # 신고·예방·치료처럼 보호 목적일 수 있는 문장은 키워드만으로 차단하지 않는다.
    if re.search(_PROTECTIVE_CONTEXT, lowered) and any(
        re.search(pat, lowered) or re.search(pat, compact) for pat in HARM_HINTS
    ):
        return "normal", False

    for pat in ABUSE_PATTERNS:
        if re.search(pat, lowered) or re.search(pat, compact):
            return "abuse_request", True
    for pat in INJECTION_PATTERNS:
        if re.search(pat, lowered):
            return "injection", True
    for pat in OFF_TOPIC_PATTERNS:
        if re.search(pat, lowered):
            return "off_topic", True
    if any(re.search(pat, lowered) for pat in ROLE_HINTS):
        # 모델이 최종 판단하되, 호출이 실패하면 injection 으로 막는다(fail-closed).
        return "injection", False
    if any(re.search(pat, lowered) or re.search(pat, compact) for pat in HARM_HINTS):
        # 가해·거래를 암시하는 표현이 있으면 온토픽 지름길을 쓰지 않는다.
        return "abuse_request", False
    if any(word in lowered for word in ON_TOPIC):
        # 도메인 단어만으로 안전을 확정하면 새로운 위해 표현이 그대로 통과한다.
        # 정상 후보로 표시하되 반드시 의미 분류기를 거친다.
        return "normal", False
    return "off_topic", False  # 애매하므로 모델에게 확인시킨다


def block_message(label: str, text: str = "") -> str:
    """차단 사유에 맞는 안내문을 고른다. 학대와 거래는 안내가 달라야 한다."""
    if label == "abuse_request":
        trade = re.search(
            r"(판매|팔(?:아|려고|면|고)|되팔|재판매|전매|양도|넘기|보내|거래|경매|사례비|수익|장사)",
            normalize_for_screening(text),
        )
        return TRADE_MESSAGE if trade else HARM_MESSAGE
    return BLOCK_MESSAGE.get(label, BLOCK_MESSAGE["off_topic"])


def mask_pii(text: str) -> str:
    text = MOBILE_RE.sub("010-****-****", text or "")
    return EMAIL_RE.sub("***@***", text)
