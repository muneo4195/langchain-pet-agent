"""설계서 3.1 — 앱 레벨 캐시의 지역명 정규화."""

import pytest

from petpal.codes import CodeTables, RegionMatch, norm_sgg, norm_sido


@pytest.mark.parametrize("raw,expected", [
    ("서울특별시", "서울"), ("서울시", "서울"), ("서울", "서울"),
    ("강원특별자치도", "강원"), ("경기도", "경기"),
    ("충남", "충청남"), ("전북", "전라북"), ("제주특별자치도", "제주"),
])
def test_norm_sido(raw, expected):
    assert norm_sido(raw) == expected


@pytest.mark.parametrize("raw,expected", [("강릉시", "강릉"), ("가평군", "가평"), ("마포구", "마포"), ("중구", "중구")])
def test_norm_sgg(raw, expected):
    assert norm_sgg(raw) == expected


@pytest.fixture
def tables():
    """실제 코드 체계를 축소한 표. 두 API 의 자릿수가 다른 점이 핵심이다."""
    return CodeTables(
        animal_sido={"서울": "6110000", "강원": "6530000", "경기": "6410000"},
        animal_sigungu={"6110000": {"마포": "3130000"}, "6530000": {"강릉": "4201000"},
                        "6410000": {"가평": "4160000"}},
        travel_regn={"서울": "11", "강원": "51", "경기": "41"},
        travel_signgu={"11": {"마포": "440"}, "51": {"강릉": "150"}, "41": {"가평": "820"}},
    )


def test_resolve_full_name(tables):
    m = tables.resolve("서울 마포구")
    assert (m.upr_cd, m.org_cd) == ("6110000", "3130000")
    assert (m.l_dong_regn_cd, m.l_dong_signgu_cd) == ("11", "440")


def test_resolve_sigungu_only(tables):
    """'강릉' 처럼 시도를 생략해도 소속 시도를 되짚는다."""
    m = tables.resolve("강릉")
    assert m.sido == "강원" and m.l_dong_signgu_cd == "150"


def test_resolve_unknown_is_empty(tables):
    assert tables.resolve("없는지명").is_empty()
    assert RegionMatch().is_empty()
