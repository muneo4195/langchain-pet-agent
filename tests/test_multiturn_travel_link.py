"""S-04 설계서 테스트 시나리오 — Multi-turn 연계 검색.

'관심동물 선택 후 해당 지역 기준 동반여행지 연계 검색' 전체를 검증한다.
test_middleware.py::test_region_falls_back_to_selected_animal 은 지역 폴백 하나만 단위로
확인하므로, 여기서는 그 뒤로 이어지는 실제 여행지 검색·동반조건 자동 보강까지 한 번에 묶어
'연계'가 실제로 동작함을 확인한다. LLM 호출 없이 Tool + Middleware 만으로 검증한다.
"""

import json
from types import SimpleNamespace

from langchain.agents.middleware import ToolCallRequest
from langgraph.types import Command

from petpal import tools
from petpal.codes import CodeTables
from petpal.middleware import region_code_resolver, result_filter, tool_cache
from petpal.tools import ALL_TOOLS

TOOLS = {t.name: t for t in ALL_TOOLS}
CHAIN = [result_filter, region_code_resolver, tool_cache]


class FakeClient:
    def __init__(self):
        self.area_based_calls = []

    def area_based_list(self, **kwargs):
        self.area_based_calls.append(kwargs)
        return [
            {"contentid": "C1", "title": "강릉 반려동물 동반 카페"},
            {"contentid": "C2", "title": "강릉 반려동물 동반 펜션"},
        ]

    def detail_pet_tour(self, content_id):
        if content_id == "C1":
            return {"acmpyTypeCd": "전구역 동반가능", "acmpyPsblCpam": "소형견"}
        return {}


def make_services():
    tables = CodeTables(
        animal_sido={"강원": "6530000"},
        animal_sigungu={"6530000": {"강릉": "4201000"}},
        travel_regn={"강원": "51"},
        travel_signgu={"51": {"강릉": "150"}},
    )
    return SimpleNamespace(client=FakeClient(), codes=tables, settings=SimpleNamespace(detail_fanout=5))


def run_tool(name, args, state):
    def invoke(request):
        return TOOLS[request.tool_call["name"]].invoke(request.tool_call)

    handler = invoke
    for mw in reversed(CHAIN):
        handler = (lambda mw, nxt: lambda req: mw.wrap_tool_call(req, nxt))(mw, handler)

    request = ToolCallRequest(
        tool_call={"name": name, "args": args, "id": "call-1", "type": "tool_call"},
        tool=TOOLS[name], state=state, runtime=None,
    )
    result = handler(request)
    if isinstance(result, Command):
        message = result.update["messages"][0]
        update = {k: v for k, v in result.update.items() if k != "messages"}
    else:
        message, update = result, {}
    return json.loads(message.content), update


def test_selecting_animal_then_nearby_travel_resolves_region_and_enriches_companion_info(monkeypatch):
    """1턴에서 고른 동물의 지역을 2턴 여행지 검색이 이어받아, 동반조건까지 자동으로 붙는지 확인."""
    services = make_services()
    monkeypatch.setattr(tools, "get_services", lambda: services)

    # 1턴: 검색 결과 중 세 번째 동물을 골랐다고 가정 — 이후 State 에 관심동물로 남는다.
    state = {"messages": [], "selected_animal": {"orgNm": "강원도 강릉시"}}

    # 2턴: "그 근처에 반려동물 동반 가능한 곳 있어?" — 지역을 생략한 지시어성 요청을 흉내낸다.
    body, update = run_tool("search_pet_friendly_travel", {"region": ""}, state)
    state.update(update)

    call = services.client.area_based_calls[0]
    assert call["l_dong_regn_cd"] == "51" and call["l_dong_signgu_cd"] == "150", \
        "관심동물의 지역(강원도 강릉시)이 여행지 검색 API 코드로 이어져야 한다"

    ids = [row["contentid"] for row in body["items"]]
    assert ids == ["C1", "C2"]
    assert body["items"][0]["acmpy_type"] == "전구역 동반가능"
    assert body["items"][0]["allowed_species"] == "소형견"
    assert body["items"][1]["acmpy_type"] == "미확인", "동반조건이 없는 장소는 지어내지 않고 미확인으로 남긴다"


def test_without_selected_animal_region_is_not_guessed(monkeypatch):
    """관심동물이 없으면 지역을 함부로 추측하지 않고 region_unresolved 를 돌려준다."""
    services = make_services()
    monkeypatch.setattr(tools, "get_services", lambda: services)

    state = {"messages": []}
    body, _ = run_tool("search_pet_friendly_travel", {"region": ""}, state)

    assert body["error"] == "region_unresolved"
    assert services.client.area_based_calls == []
