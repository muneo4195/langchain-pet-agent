"""공공 API 응답 필드 파싱과 후처리 필터 규칙.

구조동물 API 는 나이·성별·체중을 검색 파라미터로 받지 않고 응답 필드로만 주므로
(설계서 1.2 / 2.5 참조) 여기 규칙으로 조회 후 걸러낸다.
"""

from __future__ import annotations

import re
from datetime import date, datetime

# weight "0.7(Kg)" / age "2026(년생)" 처럼 단위가 붙은 문자열로 온다.
_NUM = re.compile(r"-?\d+(?:\.\d+)?")

SIZE_BOUNDS = {"소형견": (0.0, 10.0), "중형견": (10.0, 25.0), "대형견": (25.0, 1_000.0)}
SIZE_ALIASES = {
    "소형": "소형견", "소형견": "소형견", "작은": "소형견",
    "중형": "중형견", "중형견": "중형견",
    "대형": "대형견", "대형견": "대형견", "큰": "대형견",
}


def parse_number(raw: object) -> float | None:
    """'0.7(Kg)' → 0.7 / '2026(년생)' → 2026.0 / 빈 값 → None."""
    if raw is None:
        return None
    m = _NUM.search(str(raw))
    return float(m.group()) if m else None


def parse_weight_kg(raw: object) -> float | None:
    return parse_number(raw)


def parse_birth_year(raw: object) -> int | None:
    year = parse_number(raw)
    return int(year) if year and 1900 < year < 2200 else None


def age_years(raw: object, today: date | None = None) -> int | None:
    """'2023(년생)' → 만 나이 근사값."""
    birth = parse_birth_year(raw)
    if birth is None:
        return None
    return max(0, (today or date.today()).year - birth)


def size_of(weight_kg: float | None) -> str | None:
    if weight_kg is None:
        return None
    for name, (lo, hi) in SIZE_BOUNDS.items():
        if lo <= weight_kg < hi:
            return name
    return None


def normalize_size(text: str | None) -> str | None:
    if not text:
        return None
    for alias, canonical in SIZE_ALIASES.items():
        if alias in text:
            return canonical
    return None


def is_closed_notice(process_state: object) -> bool:
    """processState 가 '종료(자연사)' 처럼 종결 상태인지. 설계서 3.3 종결·비활성 필터."""
    return str(process_state or "").strip().startswith("종료")


def urgency_of(notice_edt: object, today: date | None = None) -> str | None:
    """noticeEdt(YYYYMMDD) 기준 공고 마감 임박도. 2일 이내 high / 5일 이내 medium."""
    raw = str(notice_edt or "").strip()
    if not re.fullmatch(r"\d{8}", raw):
        return None
    end = datetime.strptime(raw, "%Y%m%d").date()
    left = (end - (today or date.today())).days
    if left < 0:
        return None
    if left <= 2:
        return "high"
    if left <= 5:
        return "medium"
    return "low"


def match_score(item: dict, wanted: dict) -> tuple[float, str]:
    """규칙 기반 적합도. 모델이 임의로 점수를 매기지 않도록 시스템이 계산한다.

    (일치한 조건 수 / 요청한 조건 수) 로 계산하고 근거 문구를 함께 돌려준다.
    """
    checks: list[tuple[bool, str]] = []

    if wanted.get("size"):
        got = size_of(parse_weight_kg(item.get("weight")))
        checks.append((got == wanted["size"], f"{wanted['size']}"))
    if wanted.get("upkind"):
        checks.append((item.get("upKindCd") == wanted["upkind"], item.get("upKindNm") or "축종"))
    if wanted.get("kind_name"):
        checks.append((wanted["kind_name"] in (item.get("kindNm") or ""), wanted["kind_name"]))
    if wanted.get("sex"):
        checks.append((item.get("sexCd") == wanted["sex"], "성별"))
    if wanted.get("region_name"):
        blob = f"{item.get('orgNm', '')} {item.get('careAddr', '')} {item.get('happenPlace', '')}"
        checks.append((wanted["region_name"] in blob, wanted["region_name"]))
    if wanted.get("max_age") is not None:
        got = age_years(item.get("age"))
        checks.append((got is not None and got <= wanted["max_age"], f"{wanted['max_age']}살 이하"))

    if not checks:
        return 0.5, "조건이 지정되지 않아 최신 공고 순으로 제시합니다"

    hit = [label for ok, label in checks if ok]
    score = round(len(hit) / len(checks), 2)
    reason = f"요청 조건 {len(checks)}개 중 {len(hit)}개 일치" + (f" ({', '.join(hit)})" if hit else "")
    return score, reason[:100]
