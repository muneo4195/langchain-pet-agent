"""공공 API 실호출 검증. 네트워크가 필요하며 일일 트래픽을 소모한다.

    pytest tests/test_live_api.py -m live
기본 실행에서는 제외된다.
"""

import os

import pytest

pytestmark = pytest.mark.live

pytest.importorskip("requests")


@pytest.fixture(scope="module")
def svc():
    if not os.environ.get("DATA_GO_KR_SERVICE_KEY", "").strip() or \
       os.environ.get("DATA_GO_KR_SERVICE_KEY") == "test-key":
        pytest.skip("DATA_GO_KR_SERVICE_KEY 가 없어 건너뜁니다")
    from petpal.services import Services
    return Services()


def test_animal_search_returns_expected_fields(svc):
    rows = svc.client.abandonment_public(upr_cd="6110000", upkind="417000", num_of_rows=3)
    assert rows, "최근 30일 서울 개 공고가 0건일 수 있습니다"
    row = rows[0]
    for field in ("desertionNo", "kindNm", "weight", "age", "processState", "careTel"):
        assert field in row, f"{field} 필드가 응답에 없습니다"


def test_travel_list_has_no_companion_fields(svc):
    """설계 근거: 목록 API 에는 동반 조건이 없어 상세 조회가 반드시 필요하다."""
    rows = svc.client.area_based_list(l_dong_regn_cd="51", l_dong_signgu_cd="150",
                                      content_type_id="32", num_of_rows=3)
    assert rows
    assert "acmpyTypeCd" not in rows[0]

    detail = svc.client.detail_pet_tour(rows[0]["contentid"])
    assert "acmpyTypeCd" in detail


def test_region_codes_differ_between_apis(svc):
    m = svc.codes.resolve("강원도 강릉시")
    assert len(m.upr_cd) == 7 and len(m.l_dong_regn_cd) == 2 and len(m.l_dong_signgu_cd) == 3
