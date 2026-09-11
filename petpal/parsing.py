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


def care_duration_days(happen_dt: object, today: date | None = None) -> int | None:
    """happenDt(YYYYMMDD, 발견/구조일) 기준 보호 경과일. 장기보호 가산점 계산에 쓴다."""
    raw = str(happen_dt or "").strip()
    if not re.fullmatch(r"\d{8}", raw):
        return None
    started = datetime.strptime(raw, "%Y%m%d").date()
    days = ((today or date.today()) - started).days
    return days if days >= 0 else None


# 장기보호 가산점 — 60일부터 시작해 1년(365일)에서 상한. 조건 매칭 점수를 뒤집을 만큼 크면
# "적합도"의 의미가 흐려지므로 최대 0.1 로 작게 둔다(선호조건 하나 가중치의 1/10 수준).
_LONG_CARE_MIN_DAYS = 60
_LONG_CARE_MAX_DAYS = 365
_LONG_CARE_MAX_BONUS = 0.1


def _long_care_bonus(item: dict, today: date | None = None) -> float:
    days = care_duration_days(item.get("happenDt"), today)
    if days is None or days < _LONG_CARE_MIN_DAYS:
        return 0.0
    span = min(days, _LONG_CARE_MAX_DAYS) - _LONG_CARE_MIN_DAYS
    return round(_LONG_CARE_MAX_BONUS * span / (_LONG_CARE_MAX_DAYS - _LONG_CARE_MIN_DAYS), 3)


# 공고 특이사항(specialMark)은 자유 텍스트라 정확한 필터가 불가능하다.
# 대신 '언급이 있는 경우에만' 신호로 삼아 보수적으로 반영한다.
_GENTLE = [r"순하", r"온순", r"얌전", r"사람\s*좋아", r"친화", r"애교"]
_LOW_ACTIVITY = [r"활동량.{0,4}(적|낮)", r"조용", r"차분", r"실내\s*위주", r"산책.{0,4}적"]
_APARTMENT = [r"아파트", r"실내", r"실내견", r"집\s*에서", r"실내\s*생활"]

# 필수(스펙) 조건과 선호(추정) 조건의 가중치를 다르게 둔다.
# 크기·축종·품종·성별·지역·나이는 사실상 스펙이라 안 맞으면 후보 가치가 크게 떨어지고,
# 성격·활동량·중성화 여부는 specialMark 기반 추정이거나 상대적으로 유연한 선호라 가볍게 반영한다.
_HARD_WEIGHT = 2
_SOFT_WEIGHT = 1


def match_score(item: dict, wanted: dict, *, today: date | None = None) -> tuple[float, str]:
    """규칙 기반 적합도. 모델이 임의로 점수를 매기지 않도록 시스템이 계산한다.

    필수(스펙) 조건과 선호(추정) 조건을 가중합으로 합산하고, 장기보호 동물에는
    작은 가산점을 더한다. 조건 미지정 시 0.5 중립값은 그대로 유지한다.
    """
    special = str(item.get("specialMark") or "")
    blob = f"{item.get('orgNm', '')} {item.get('careAddr', '')} {item.get('happenPlace', '')}"

    def _has_any(patterns: list[str], text: str) -> bool:
        return any(re.search(p, text) for p in patterns)

    hard_checks: list[tuple[bool, str]] = []
    soft_checks: list[tuple[bool, str]] = []

    if wanted.get("size"):
        got = size_of(parse_weight_kg(item.get("weight")))
        hard_checks.append((got == wanted["size"], f"{wanted['size']}"))
    if wanted.get("upkind"):
        hard_checks.append((item.get("upKindCd") == wanted["upkind"], item.get("upKindNm") or "축종"))
    if wanted.get("kind_name"):
        hard_checks.append((wanted["kind_name"] in (item.get("kindNm") or ""), wanted["kind_name"]))
    if wanted.get("sex"):
        hard_checks.append((item.get("sexCd") == wanted["sex"], "성별"))
    if wanted.get("region_name"):
        hard_checks.append((wanted["region_name"] in blob, wanted["region_name"]))
    if wanted.get("max_age") is not None:
        got = age_years(item.get("age"))
        hard_checks.append((got is not None and got <= wanted["max_age"], f"{wanted['max_age']}살 이하"))
    if wanted.get("min_age") is not None:
        got = age_years(item.get("age"))
        hard_checks.append((got is not None and got >= wanted["min_age"], f"{wanted['min_age']}살 이상"))
    if wanted.get("gentle"):
        soft_checks.append((_has_any(_GENTLE, special), "순한(특이사항)"))
    if wanted.get("low_activity"):
        soft_checks.append((_has_any(_LOW_ACTIVITY, special), "저활동(특이사항)"))
    if wanted.get("apartment"):
        soft_checks.append((_has_any(_APARTMENT, special), "아파트/실내(특이사항)"))
    if wanted.get("neutered"):
        soft_checks.append((item.get("neuterYn") == "Y", "중성화 완료"))

    checks = hard_checks + soft_checks
    bonus = _long_care_bonus(item, today)

    if not checks:
        score = round(min(1.0, 0.5 + bonus), 2)
        reason = "조건이 지정되지 않아 최신 공고 순으로 제시합니다"
    else:
        hit = [label for ok, label in checks if ok]
        total_weight = _HARD_WEIGHT * len(hard_checks) + _SOFT_WEIGHT * len(soft_checks)
        hit_weight = (
            _HARD_WEIGHT * sum(1 for ok, _ in hard_checks if ok)
            + _SOFT_WEIGHT * sum(1 for ok, _ in soft_checks if ok)
        )
        score = round(min(1.0, hit_weight / total_weight + bonus), 2)
        reason = f"요청 조건 {len(checks)}개 중 {len(hit)}개 일치" + (f" ({', '.join(hit)})" if hit else "")
        # 특이사항 기반 조건은 '언급이 없는 경우'가 많아, 근거가 없을 때는 미확인을 분명히 남긴다.
        wants_traits = any(wanted.get(k) for k in ("gentle", "low_activity", "apartment"))
        hit_trait = any("특이사항" in h for h in hit)
        if wants_traits and not hit_trait:
            reason += " / 특이사항 없음" if not special.strip() else " / 특이사항 근거 없음"

    if bonus > 0:
        days = care_duration_days(item.get("happenDt"), today)
        reason += f" · 장기보호 {days}일째"
    return score, reason[:100]
