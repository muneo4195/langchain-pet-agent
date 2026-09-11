"""S-01 설계서 테스트 시나리오 — Happy Path.

'지역+축종 조건으로 구조동물 검색 → 체중 기준 후처리 → 상태 조회' 전체 흐름을
scripts/demo_pipeline.py 와 동일한 미들웨어 체인으로 자동화된 회귀 테스트로 옮긴 것.
LLM 호출 없이 Tool + Middleware 만으로 검증한다.
"""

import json
from types import SimpleNamespace

from langchain.agents.middleware import ToolCallRequest
from langchain.messages import HumanMessage
from langchain.tools import ToolRuntime
from langgraph.types import Command

from petpal import tools
from petpal.codes import CodeTables
from petpal.middleware import region_code_resolver, result_filter, tool_cache
from petpal.tools import ALL_TOOLS

TOOLS = {t.name: t for t in ALL_TOOLS}
# agent.py 주석의 실제 중첩 순서: result_filter(바깥) → region_code_resolver → tool_cache(안쪽) → Tool
CHAIN = [result_filter, region_code_resolver, tool_cache]
NEEDS_RUNTIME = {"get_animal_detail", "save_user_preference"}

ROWS = [
    {"desertionNo": "A1", "weight": "3(Kg)", "age": "2024(년생)", "processState": "보호중",
     "upKindCd": "417000", "upKindNm": "개", "kindNm": "푸들", "noticeEdt": "20991231",
     "orgNm": "서울특별시", "careNm": "서울보호소", "careTel": "02-000-0000"},
    {"desertionNo": "A2", "weight": "20(Kg)", "age": "2020(년생)", "processState": "종료(입양)",
     "upKindCd": "417000", "upKindNm": "개", "kindNm": "리트리버", "noticeEdt": "20991231",
     "orgNm": "서울특별시"},
]


class FakeClient:
    def __init__(self):
        self.search_calls = []

    def abandonment_public(self, **kwargs):
        self.search_calls.append(kwargs)
        return [dict(r) for r in ROWS]


def make_services():
    tables = CodeTables(animal_sido={"서울": "6110000"})
    return SimpleNamespace(client=FakeClient(), codes=tables, settings=SimpleNamespace(detail_fanout=5))


def run_tool(name, args, state):
    def invoke(request):
        call = request.tool_call
        call_args = dict(call["args"])
        if call["name"] in NEEDS_RUNTIME:
            call_args["runtime"] = ToolRuntime(
                state=request.state, context=None, config={},
                stream_writer=lambda _: None, tool_call_id=call["id"], store=None,
            )
        return TOOLS[call["name"]].invoke({**call, "args": call_args})

    handler = invoke
    for mw in reversed(CHAIN):
        handler = (lambda mw, nxt: lambda req: mw.wrap_tool_call(req, nxt))(mw, handler)

    request = ToolCallRequest(
        tool_call={"name": name, "args": args, "id": f"call-{name}", "type": "tool_call"},
        tool=TOOLS[name], state=state, runtime=None,
    )
    result = handler(request)
    if isinstance(result, Command):
        message = result.update["messages"][0]
        update = {k: v for k, v in result.update.items() if k != "messages"}
    else:
        message, update = result, {}
    return json.loads(message.content), update


def test_region_and_species_search_then_weight_filter_then_status_query(monkeypatch):
    """S-01: 검색 → 소형견만 남기는 체중 후처리 → 남은 공고의 상태(processState) 조회까지 한 번에 확인."""
    services = make_services()
    monkeypatch.setattr(tools, "get_services", lambda: services)

    state = {"messages": [HumanMessage("서울에 있는 순한 소형견 강아지 보여줘")]}

    body, update = run_tool("search_rescued_animals", {"region": "서울", "upkind": "개"}, state)
    state.update(update)

    assert services.client.search_calls[0]["upr_cd"] == "6110000", "지역명이 구조동물 API 코드로 변환돼야 한다"
    assert body["region"] == "서울"
    ids = [r["desertionNo"] for r in body["items"]]
    assert ids == ["A1"], "체중 기반 크기 후처리로 소형견(A1)만 남고 대형견(A2)은 제외돼야 한다"

    # 이어서 방금 찾은 공고의 상태를 조회한다 — API 재호출 없이 직전 결과 캐시에서 나와야 한다.
    detail, update2 = run_tool("get_animal_detail", {"desertion_no": "A1"}, state)
    state.update(update2)

    assert detail["processState"] == "보호중"
    assert detail["careTel"] == "02-000-0000"
    assert state["selected_animal"]["desertionNo"] == "A1"


def test_zero_result_after_weight_filter_is_reported_not_fabricated(monkeypatch):
    """체중 후처리로 전부 걸러지면 0건으로 정직하게 보고하고 대안을 제시한다(1.4 DoD)."""
    services = make_services()
    monkeypatch.setattr(tools, "get_services", lambda: services)

    state = {"messages": [HumanMessage("서울에 있는 대형견만 보여줘")]}
    body, _ = run_tool("search_rescued_animals", {"region": "서울", "upkind": "개"}, state)

    # ROWS 에는 대형견이 없다(중형/소형만 존재) — 조건에 맞는 결과가 없어야 한다.
    assert body["items"] == []
    assert body["no_result_hint"]
