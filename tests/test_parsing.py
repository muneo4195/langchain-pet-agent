"""설계서 2.5 — API 응답 필드 후처리 규칙."""

from datetime import date

import pytest

from petpal import parsing as p


@pytest.mark.parametrize("raw,expected", [("0.7(Kg)", 0.7), ("3(Kg)", 3.0), ("", None), (None, None)])
def test_parse_weight(raw, expected):
    assert p.parse_weight_kg(raw) == expected


def test_age_years():
    assert p.age_years("2023(년생)", today=date(2026, 9, 10)) == 3
    assert p.age_years("미상") is None


@pytest.mark.parametrize("kg,size", [(3.2, "소형견"), (10.0, "중형견"), (24.9, "중형견"), (30, "대형견"), (None, None)])
def test_size_of(kg, size):
    assert p.size_of(kg) == size


def test_normalize_size_from_utterance():
    assert p.normalize_size("아파트에서 키우기 좋은 순한 소형견") == "소형견"
    assert p.normalize_size("중형견 입장 가능한 곳") == "중형견"
    assert p.normalize_size("아무거나") is None


@pytest.mark.parametrize("state,closed", [
    ("종료(자연사)", True), ("종료(입양)", True), ("보호중", False), ("공고중", False), ("", False),
])
def test_is_closed_notice(state, closed):
    assert p.is_closed_notice(state) is closed


def test_urgency():
    today = date(2026, 9, 10)
    assert p.urgency_of("20260911", today) == "high"
    assert p.urgency_of("20260914", today) == "medium"
    assert p.urgency_of("20260930", today) == "low"
    assert p.urgency_of("20260901", today) is None  # 이미 지난 공고
    assert p.urgency_of("", today) is None


def test_match_score_is_rule_based():
    """설계서 2.3 제약: 적합도는 모델이 아니라 시스템이 계산한다."""
    item = {"weight": "3(Kg)", "upKindCd": "417000", "upKindNm": "개", "orgNm": "서울특별시"}
    score, reason = p.match_score(item, {"size": "소형견", "upkind": "417000", "region_name": "서울"})
    assert score == 1.0 and "3개 일치" in reason

    score, _ = p.match_score(item, {"size": "대형견", "upkind": "417000"})
    assert score == 0.5
    assert p.match_score(item, {})[0] == 0.5  # 조건이 없으면 중립값


def test_match_score_can_use_special_mark_traits_conservatively():
    item = {
        "weight": "3(Kg)",
        "upKindCd": "417000",
        "upKindNm": "개",
        "orgNm": "서울특별시",
        "specialMark": "온순하고 활동량 적음. 실내 생활 가능",
    }
    wanted = {"size": "소형견", "gentle": True, "low_activity": True, "apartment": True}
    score, reason = p.match_score(item, wanted)
    assert score == 1.0
    assert "순한(특이사항)" in reason
    assert "저활동(특이사항)" in reason
