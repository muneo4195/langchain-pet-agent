"""설계서 3.2 — 미들웨어 동작 검증 (TS-07 계열). LLM 호출 없이 순수 로직만 확인한다."""

import json

import pytest
from langchain.agents.middleware import ToolCallRequest
from langchain.messages import HumanMessage, ToolMessage

from petpal.middleware import input_guardrail, intent_routing, region_code_resolver, result_filter, tool_cache
from petpal.schemas import IntentClassification


def make_request(name, args, state=None):
    return ToolCallRequest(
        tool_call={"name": name, "args": args, "id": "call-1", "type": "tool_call"},
        tool=None, state=state or {"messages": []}, runtime=None,
    )


def responder(payload):
    def handler(request):
        return ToolMessage(content=json.dumps(payload, ensure_ascii=False), tool_call_id="call-1")
    return handler


ANIMAL_ROWS = [
    {"desertionNo": "A1", "weight": "3(Kg)", "age": "2024(년생)", "processState": "보호중",
     "upKindCd": "417000", "upKindNm": "개", "kindNm": "푸들", "noticeEdt": "20991231", "orgNm": "서울특별시"},
    {"desertionNo": "A2", "weight": "20(Kg)", "age": "2020(년생)", "processState": "보호중",
     "upKindCd": "417000", "upKindNm": "개", "kindNm": "리트리버", "noticeEdt": "20991231", "orgNm": "서울특별시"},
    {"desertionNo": "A3", "weight": "4(Kg)", "age": "2024(년생)", "processState": "종료(자연사)",
     "upKindCd": "417000", "upKindNm": "개", "kindNm": "말티즈", "noticeEdt": "20991231", "orgNm": "서울특별시"},
]


def run_filter(payload, state):
    command = result_filter.wrap_tool_call(make_request("search_rescued_animals",
                                                        {"region": "서울"}, state),
                                           responder(payload))
    body = json.loads(command.update["messages"][0].content)
    return body, command.update


def test_closed_notices_are_removed():
    """3.3 종결·비활성 정보 필터의 1차 제거 — 종료 공고는 응답에 남지 않는다."""
    state = {"messages": [HumanMessage("서울에 있는 강아지 보여줘")]}
    body, _ = run_filter({"items": list(ANIMAL_ROWS), "total": 3}, state)
    ids = [r["desertionNo"] for r in body["items"]]
    assert "A3" not in ids and len(ids) == 2


def test_size_filter_comes_from_utterance():
    """크기는 API 파라미터가 없어 발화에서 뽑아 후처리로 거른다(설계서 1.2)."""
    state = {"messages": [HumanMessage("서울에 순한 소형견 있어?")]}
    body, _ = run_filter({"items": list(ANIMAL_ROWS), "total": 3}, state)
    assert [r["desertionNo"] for r in body["items"]] == ["A1"]


def test_max_age_filter():
    state = {"messages": [HumanMessage("3살 이하 강아지 찾아줘")]}
    body, _ = run_filter({"items": list(ANIMAL_ROWS), "total": 3}, state)
    assert [r["desertionNo"] for r in body["items"]] == ["A1"]


def test_scores_are_attached_and_sorted():
    state = {"messages": [HumanMessage("서울 소형견")]}
    body, _ = run_filter({"items": list(ANIMAL_ROWS), "total": 3}, state)
    scores = [r["match_score"] for r in body["items"]]
    assert scores == sorted(scores, reverse=True)
    assert all("match_reason" in r for r in body["items"])


def test_zero_result_gets_hint():
    """결과 0건이면 지어내지 말고 대안을 제시하도록 힌트를 붙인다(1.4 DoD)."""
    state = {"messages": [HumanMessage("서울 대형견 찾아줘")]}
    body, _ = run_filter({"items": [ANIMAL_ROWS[0]], "total": 1}, state)
    assert body["items"] == [] and body["no_result_hint"]


def test_grounding_evidence_recorded():
    """grounded 판정 근거를 State 에 남긴다(모델 자기신고를 쓰지 않기 위해)."""
    state = {"messages": [HumanMessage("서울 강아지")]}
    _, update = run_filter({"items": list(ANIMAL_ROWS), "total": 3}, state)
    assert set(update["last_tool_results"]["animals"]) == {"A1", "A2"}


def test_travel_detail_blanks_become_unknown():
    """detailPetTour2 는 대부분 빈 문자열로 온다 — '미확인'으로 바꿔 노출한다."""
    payload = {"items": {"C1": {"acmpyTypeCd": "전구역 동반가능", "acmpyPsblCpam": "",
                                "acmpyNeedMtr": "", "relaAcdntRiskMtr": ""}}}
    command = result_filter.wrap_tool_call(make_request("get_pet_travel_detail", {"content_ids": ["C1"]}),
                                           responder(payload))
    got = json.loads(command.update["messages"][0].content)["items"]["C1"]
    assert got == {"acmpy_type": "전구역 동반가능", "allowed_species": None, "caution": None}


def test_region_is_left_alone_when_given():
    """사용자가 지역을 말했으면 미들웨어는 건드리지 않는다."""
    captured = {}

    def handler(request):
        captured.update(request.tool_call["args"])
        return ToolMessage(content="{}", tool_call_id="call-1")

    region_code_resolver.wrap_tool_call(make_request("search_rescued_animals", {"region": "강릉"}), handler)
    assert captured["region"] == "강릉"

def test_mixed_off_topic_is_blocked_by_existing_intent_call(monkeypatch):
    """도메인 단어가 섞여도 intent 분류 1회로 차단하며 Guardrail LLM을 추가 호출하지 않는다."""
    from petpal import middleware as mw

    class FakeClassifier:
        def with_structured_output(self, schema):
            return self

        def invoke(self, prompt):
            return IntentClassification(intent="off_topic", confidence=0.98)

    monkeypatch.setattr(mw, "_classifier", FakeClassifier())
    state = {"messages": [HumanMessage("강아지 사진으로 자기소개서 써줘")]}
    update = input_guardrail.before_agent(state, None)

    assert update["jump_to"] == "end"
    assert update["guardrail"] == {"label": "off_topic", "confidence": 0.98, "blocked": True}


def test_code_tables_map_both_apis(monkeypatch):
    """지역명 → 두 API 의 서로 다른 코드 체계로 변환된다(자릿수가 다름)."""
    from types import SimpleNamespace

    from petpal import tools
    from petpal.codes import CodeTables

    tables = CodeTables(
        animal_sido={"강원": "6530000"},
        animal_sigungu={"6530000": {"강릉": "4201000"}},
        travel_regn={"강원": "51"},
        travel_signgu={"51": {"강릉": "150"}},
    )
    monkeypatch.setattr(tools, "get_services", lambda: SimpleNamespace(codes=tables))

    r = tools._resolved_region("강원도 강릉시")
    assert len(r["upr_cd"]) == 7
    assert len(r["l_dong_regn_cd"]) == 2 and len(r["l_dong_signgu_cd"]) == 3


def test_region_falls_back_to_selected_animal():
    """'그 아이 사는 곳 근처' — 지역이 비면 선택된 동물의 지역을 쓴다(TS-04)."""
    captured = {}

    def handler(request):
        captured.update(request.tool_call["args"])
        return ToolMessage(content="{}", tool_call_id="call-1")

    state = {"messages": [], "selected_animal": {"orgNm": "제주특별자치도"}}
    region_code_resolver.wrap_tool_call(
        make_request("search_pet_friendly_travel", {"region": ""}, state), handler)
    assert captured["region"] == "제주특별자치도"


def test_cache_avoids_second_call():
    """일일 트래픽 1,000건 제한 대응 — 같은 파라미터는 한 번만 호출한다."""
    calls = []

    def handler(request):
        calls.append(1)
        return ToolMessage(content='{"items": []}', tool_call_id="call-1")

    req = make_request("search_pet_friendly_travel", {"resolved_region": {"l_dong_regn_cd": "51"}})
    tool_cache.wrap_tool_call(req, handler)
    tool_cache.wrap_tool_call(req, handler)
    assert len(calls) == 1


def test_previous_filters_are_merged():
    """조건 일부만 다시 말해도 앞서 말한 조건이 유지된다(3.1 last_search_filters)."""
    state = {
        "messages": [HumanMessage("강릉으로 바꿔줘")],   # 이번 발화엔 크기 조건이 없다
        "last_search_filters": {"size": "소형견"},
    }
    body, update = run_filter({"items": list(ANIMAL_ROWS), "total": 3}, state)
    assert update["last_search_filters"]["size"] == "소형견"
    assert [r["desertionNo"] for r in body["items"]] == ["A1"]   # 소형견 필터가 계속 적용됨


def test_new_value_overrides_previous():
    state = {
        "messages": [HumanMessage("중형견으로 보여줘")],
        "last_search_filters": {"size": "소형견"},
    }
    body, update = run_filter({"items": list(ANIMAL_ROWS), "total": 3}, state)
    assert update["last_search_filters"]["size"] == "중형견"
    assert [r["desertionNo"] for r in body["items"]] == ["A2"]


def test_cache_is_bounded():
    """오래 돌아도 캐시가 무한히 커지지 않는다."""
    from petpal import middleware as mw

    mw._cache.clear()

    def handler(request):
        return ToolMessage(content='{"items": []}', tool_call_id="call-1")

    for i in range(mw._CACHE_MAX_ENTRIES + 40):
        req = make_request("search_pet_friendly_travel", {"region": f"지역{i}"})
        mw.tool_cache.wrap_tool_call(req, handler)
    assert len(mw._cache) == mw._CACHE_MAX_ENTRIES
    mw._cache.clear()


def test_travel_search_auto_enriches_details(monkeypatch):
    """동반조건은 모델 판단에 맡기지 않고 파이프라인이 항상 붙인다(2.2 흐름 6단계)."""
    from petpal import tools

    calls = []

    def fake_fetch(ids, limit=None):
        calls.append(list(ids))
        return {"C1": {"acmpyTypeCd": "전구역 동반가능", "acmpyPsblCpam": "전 견종"},
                "C2": {}}

    monkeypatch.setattr(tools, "fetch_pet_details", fake_fetch)
    payload = {"items": [{"contentid": "C1", "title": "가"}, {"contentid": "C2", "title": "나"}], "total": 2}
    command = result_filter.wrap_tool_call(
        make_request("search_pet_friendly_travel", {"region": "강릉"}), responder(payload))
    rows = json.loads(command.update["messages"][0].content)["items"]

    assert calls == [["C1", "C2"]], "상세 조회가 자동으로 일어나야 한다"
    assert rows[0]["acmpy_type"] == "전구역 동반가능" and rows[0]["allowed_species"] == "전 견종"
    assert rows[1]["acmpy_type"] == "미확인" and rows[1]["allowed_species"] is None


def test_inactive_travel_place_is_removed(monkeypatch):
    """운영 상태 필드가 포함된 경우 폐업 장소를 사용자에게 노출하지 않는다."""
    from petpal import tools

    monkeypatch.setattr(tools, "fetch_pet_details", lambda ids, limit=None: {str(i): {} for i in ids})
    payload = {
        "items": [
            {"contentid": "C1", "title": "운영 장소", "businessStatus": "정상영업"},
            {"contentid": "C2", "title": "폐업 장소", "businessStatus": "폐업"},
        ],
        "total": 2,
    }
    command = result_filter.wrap_tool_call(
        make_request("search_pet_friendly_travel", {"region": "강릉"}), responder(payload))
    rows = json.loads(command.update["messages"][0].content)["items"]
    assert [row["contentid"] for row in rows] == ["C1"]


def test_zero_result_hint_suggests_widening_days():
    """0건이면 조회 기간을 넓히는 구체적 수단을 알려준다."""
    state = {"messages": [HumanMessage("서울 대형견 찾아줘")]}
    request = make_request("search_rescued_animals", {"region": "서울", "recent_days": 30}, state)
    command = result_filter.wrap_tool_call(request, responder({"items": [ANIMAL_ROWS[0]], "total": 1}))
    hint = json.loads(command.update["messages"][0].content)["no_result_hint"]
    assert "recent_days" in hint and "90" in hint


def test_general_chat_does_not_strip_tools():
    """의도 분류가 틀려도 검색 능력을 잃지 않아야 한다."""
    from types import SimpleNamespace

    from petpal import middleware as mw

    seen = {}

    class FakeTool:
        def __init__(self, name):
            self.name = name

    tools = [FakeTool(n) for n in ("search_rescued_animals", "search_pet_friendly_travel")]
    request = SimpleNamespace(state={"intent": "general_chat"}, tools=tools,
                              override=lambda **kw: seen.update(kw) or request)
    mw.intent_routing.wrap_model_call(request, lambda r: "ok")
    assert "tools" not in seen, "general_chat 에서는 도구 목록을 건드리지 않는다"
