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
    # 발화에서 자주 쓰이는 우회 표현. 정확 일치가 아니라 부분일치라
    # "작은 규모"처럼 문맥이 다른 경우까지 넓게 잡을 수 있어 규칙만으로는 한계가 있다
    # (애매하면 middleware._extract_conditions_llm 이 보조로 개입한다).
    "조그마": "소형견", "조그맣": "소형견", "초소형": "소형견", "미니": "소형견",
    "핸드백": "소형견", "품에": "소형견", "손바닥": "소형견", "아담": "소형견",
    "덩치": "대형견", "우람": "대형견", "골격이 좋": "대형견", "왕치": "대형견",
    "덩치가 큰": "대형견",
}
# normalize_size 가 None 을 반환했을 때, 애초에 크기 얘기를 한 건지(=규칙 실패) vs
# 크기 얘기 자체가 없는 건지(=조건 없음)를 구분하기 위한 신호어.
SIZE_SIGNAL_WORDS = (
    "사이즈", "크기", "몸집", "체구", "덩치", "체형",
    # "한 손에 쏙" 같은 우회 표현은 alias 사전이 못 잡을 수 있어 신호어로만 감지한다.
    "한 손", "손에", "가방", "들어갈", "쏙",
) + tuple(SIZE_ALIASES.keys())


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


def has_size_signal(text: str | None) -> bool:
    """크기 관련 표현이 있었는지(=규칙이 놓쳤어도 후속 처리가 필요한지) 판별."""
    if not text:
        return False
    return any(word in text for word in SIZE_SIGNAL_WORDS)


def is_closed_notice(process_state: object) -> bool:
    """processState 가 '종료(자연사)' 처럼 종결 상태인지. 설계서 3.3 종결·비활성 필터."""
    return str(process_state or "").strip().startswith("종료")


def is_inactive_place(item: dict) -> bool:
    """API가 운영 상태 필드를 제공한 경우 폐업·휴업·비활성 장소를 제거한다."""
    status_keys = ("businessStatus", "operStatus", "status", "processState")
    inactive = ("폐업", "휴업", "운영종료", "영업종료", "비활성", "종료")
    for key in status_keys:
        value = str(item.get(key) or "").replace(" ", "")
        if any(token in value for token in inactive):
            return True
    use_yn = str(item.get("useYn") or item.get("use_yn") or "").upper()
    return use_yn in {"N", "NO", "FALSE", "0"}


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
    special = str(item.get("specialMark") or "")
    blob = f"{item.get('orgNm', '')} {item.get('careAddr', '')} {item.get('happenPlace', '')}"

    def _has_any(patterns: list[str], text: str) -> bool:
        return any(re.search(p, text) for p in patterns)

    # 공고 특이사항(specialMark)은 자유 텍스트라 정확한 필터가 불가능하다.
    # 대신 '언급이 있는 경우에만' 신호로 삼아 보수적으로 반영한다.
    _GENTLE = [r"순하", r"온순", r"얌전", r"사람\s*좋아", r"친화", r"애교"]
    _LOW_ACTIVITY = [r"활동량.{0,4}(적|낮)", r"조용", r"차분", r"실내\s*위주", r"산책.{0,4}적"]
    _APARTMENT = [r"아파트", r"실내", r"실내견", r"집\s*에서", r"실내\s*생활"]

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
        checks.append((wanted["region_name"] in blob, wanted["region_name"]))
    if wanted.get("max_age") is not None:
        got = age_years(item.get("age"))
        checks.append((got is not None and got <= wanted["max_age"], f"{wanted['max_age']}살 이하"))
    if wanted.get("min_age") is not None:
        got = age_years(item.get("age"))
        checks.append((got is not None and got >= wanted["min_age"], f"{wanted['min_age']}살 이상"))
    if wanted.get("gentle"):
        checks.append((_has_any(_GENTLE, special), "순한(특이사항)"))
    if wanted.get("low_activity"):
        checks.append((_has_any(_LOW_ACTIVITY, special), "저활동(특이사항)"))
    if wanted.get("apartment"):
        checks.append((_has_any(_APARTMENT, special), "아파트/실내(특이사항)"))

    if not checks:
        return 0.5, "조건이 지정되지 않아 최신 공고 순으로 제시합니다"

    hit = [label for ok, label in checks if ok]
    score = round(len(hit) / len(checks), 2)
    reason = f"요청 조건 {len(checks)}개 중 {len(hit)}개 일치" + (f" ({', '.join(hit)})" if hit else "")
    # 특이사항 기반 조건은 '언급이 없는 경우'가 많아, 근거가 없을 때는 미확인을 분명히 남긴다.
    wants_traits = any(wanted.get(k) for k in ("gentle", "low_activity", "apartment"))
    hit_trait = any("특이사항" in h for h in hit)
    if wants_traits and not hit_trait:
        reason += " / 특이사항 없음" if not special.strip() else " / 특이사항 근거 없음"
    return score, reason[:100]
