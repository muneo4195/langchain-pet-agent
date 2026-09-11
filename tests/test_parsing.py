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


def test_match_score_weighs_hard_spec_conditions_more_than_soft_preferences():
    """설계서 보강: 필수(스펙) 조건 불일치가 선호 조건 불일치보다 점수에 더 크게 반영돼야 한다."""
    item = {"weight": "3(Kg)", "upKindCd": "417000", "specialMark": "온순하고 사람 좋아함"}

    # 필수(크기) 하나만 어긋난 경우
    hard_miss, _ = p.match_score(item, {"size": "대형견", "gentle": True})
    # 선호(성격) 하나만 어긋난 경우 — specialMark 에 저활동 언급이 없음
    soft_miss, _ = p.match_score(item, {"size": "소형견", "low_activity": True})

    assert soft_miss > hard_miss


def test_match_score_neutered_preference_is_soft():
    item_neutered = {"weight": "3(Kg)", "neuterYn": "Y"}
    item_not = {"weight": "3(Kg)", "neuterYn": "N"}
    wanted = {"size": "소형견", "neutered": True}

    score_y, reason_y = p.match_score(item_neutered, wanted)
    score_n, reason_n = p.match_score(item_not, wanted)

    assert score_y > score_n
    assert "중성화 완료" in reason_y
    assert "중성화 완료" not in reason_n


def test_care_duration_days():
    assert p.care_duration_days("20260101", today=date(2026, 9, 10)) == 252
    assert p.care_duration_days("미상") is None
    assert p.care_duration_days("") is None


def test_match_score_gives_small_bonus_to_long_term_care_animals():
    """설계 보강: 조건이 동률이면 오래 보호된 아이가 근소하게 더 높은 점수를 받는다.

    이미 1.0으로 만점인 후보는 보너스가 상한(1.0)에 가려 보이지 않으므로,
    조건을 지정하지 않아 0.5 중립값에서 시작하는 경우로 확인한다.
    """
    today = date(2026, 9, 10)
    long_care = {"happenDt": "20260101"}   # 252일째
    recent = {"happenDt": "20260901"}      # 9일째 — 보너스 없음

    long_score, long_reason = p.match_score(long_care, {}, today=today)
    recent_score, _ = p.match_score(recent, {}, today=today)

    assert long_score > recent_score
    assert "장기보호" in long_reason
    # 가산점이 기본 스코어링을 뒤집을 만큼 크면 안 된다.
    assert long_score - recent_score <= 0.1


def test_match_score_bonus_does_not_exceed_1_0():
    today = date(2026, 9, 10)
    item = {"weight": "3(Kg)", "happenDt": "20250101"}  # 365일 이상 — 상한 적용
    score, _ = p.match_score(item, {"size": "소형견"}, today=today)
    assert score == 1.0
