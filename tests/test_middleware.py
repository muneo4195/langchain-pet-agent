"""설계서 3.2 — 미들웨어 동작 검증 (TS-07 계열). LLM 호출 없이 순수 로직만 확인한다."""

import json
from datetime import date

import pytest
from langchain.agents.middleware import ToolCallRequest
from langchain.messages import HumanMessage, ToolMessage

from petpal.middleware import input_guardrail, intent_routing, region_code_resolver, result_filter, tool_cache
from petpal.schemas import ConditionExtraction, GuardrailClassification, IntentClassification


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


def run_filter(payload, state, args=None):
    command = result_filter.wrap_tool_call(make_request("search_rescued_animals",
                                                        {"region": "서울", **(args or {})}, state),
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


def test_size_from_model_tool_arg_handles_metaphorical_phrasing():
    """설계 보강: 은유적 표현("핸드백에 들어갈 정도")은 정규식으로 못 잡지만,
    모델이 Tool 인자(size)로 구조화해 넘기면 정상적으로 필터링된다."""
    state = {"messages": [HumanMessage("핸드백에 들어갈 정도로 조그마한 애 있을까?")]}
    # 정규식 폴백만으로는 절대 소형견을 못 뽑는 문장이라는 걸 먼저 확인한다.
    body, _ = run_filter({"items": list(ANIMAL_ROWS), "total": 3}, state)
    assert len(body["items"]) == 2  # 필터 없이 종결 공고만 제거된 상태

    body, _ = run_filter({"items": list(ANIMAL_ROWS), "total": 3}, state, args={"size": "소형견"})
    assert [r["desertionNo"] for r in body["items"]] == ["A1"]


def test_min_age_from_model_tool_arg():
    """"2살 넘은 애" 처럼 하한 나이는 정규식엔 없던 개념이라 Tool 인자로만 표현된다."""
    state = {"messages": [HumanMessage("2살 넘은 강아지 보여줘")]}
    body, _ = run_filter({"items": list(ANIMAL_ROWS), "total": 3}, state, args={"min_age": 5})
    assert [r["desertionNo"] for r in body["items"]] == ["A2"]


def test_model_tool_arg_size_beats_regex_when_both_present():
    """모델이 Tool 인자를 채웠으면 정규식 폴백보다 우선한다."""
    state = {"messages": [HumanMessage("대형견 찾아줘")]}
    body, _ = run_filter({"items": list(ANIMAL_ROWS), "total": 3}, state, args={"size": "소형견"})
    assert [r["desertionNo"] for r in body["items"]] == ["A1"]


def test_scores_are_attached_and_sorted():
    state = {"messages": [HumanMessage("서울 소형견")]}
    body, _ = run_filter({"items": list(ANIMAL_ROWS), "total": 3}, state)
    scores = [r["match_score"] for r in body["items"]]
    assert scores == sorted(scores, reverse=True)
    assert all("match_reason" in r for r in body["items"])


def test_llm_condition_extraction_fills_size_when_rule_misses(monkeypatch):
    """규칙 normalize_size가 못 잡아도 size 신호어가 있으면 LLM 보조 추출로 size를 채운다."""
    from petpal import middleware as mw

    class FakeClassifier:
        def __init__(self):
            self.schema = None

        def with_structured_output(self, schema):
            self.schema = schema
            return self

        def invoke(self, prompt):
            if self.schema is IntentClassification:
                return IntentClassification(intent="adoption_search", confidence=0.9)
            if self.schema is ConditionExtraction:
                return ConditionExtraction(size="소형견", min_age=None, max_age=None, confidence=0.9, note=None)
            raise AssertionError(f"unexpected schema: {self.schema}")

    monkeypatch.setattr(mw, "rule_screen", lambda _: ("normal", True))
    monkeypatch.setattr(mw, "_classifier", FakeClassifier())

    # '한 손에 쏙 들어가는'은 normalize_size alias에 없지만 크기 신호어로 감지되어 LLM 보조가 동작해야 한다.
    text = "서울에서 한 손에 쏙 들어가는 애로 보여줘"
    state = {"messages": [HumanMessage(text)]}
    update = input_guardrail.before_agent(state, None)
    assert update["extracted_conditions"]["size"] == "소형견"

    # 결과 필터가 이 값을 _wanted_from에서 병합해 실제 하드 필터(size)에 반영한다.
    year = date.today().year
    rows = [
        {**ANIMAL_ROWS[0], "desertionNo": "A1", "weight": "5(Kg)", "age": f"{year}(년생)"},
        {**ANIMAL_ROWS[0], "desertionNo": "A2", "weight": "20(Kg)", "age": f"{year}(년생)"},
    ]
    state2 = {**state, **update}
    body, _ = run_filter({"items": rows, "total": 2}, state2)
    assert [r["desertionNo"] for r in body["items"]] == ["A1"]


def test_rule_age_conditions_support_year_and_exclusion_phrases():
    """'2년 넘은 애는 말고' 같은 표현이 max_age로 해석되어 하드 필터에 적용된다."""
    year = date.today().year
    rows = [
        {**ANIMAL_ROWS[0], "desertionNo": "A1", "age": f"{year}(년생)"},
        {**ANIMAL_ROWS[0], "desertionNo": "A2", "age": f"{year-5}(년생)"},
    ]
    state = {"messages": [HumanMessage("서울에서 2년 넘은 애는 말고")]}
    body, update = run_filter({"items": rows, "total": 2}, state)
    assert [r["desertionNo"] for r in body["items"]] == ["A1"]
    assert update["last_search_filters"]["max_age"] == 2


def test_traits_from_special_mark_can_affect_ranking():
    """특이사항 기반(순함/저활동/실내) 조건이 있으면 match_score/정렬에 실제로 반영된다."""
    rows = [
        {**ANIMAL_ROWS[0], "desertionNo": "A1", "specialMark": "온순하고 활동량 적음. 실내 생활 가능"},
        {**ANIMAL_ROWS[0], "desertionNo": "A2", "specialMark": "특이사항 없음"},
    ]
    state = {"messages": [HumanMessage("서울 소형견. 아파트에서 키우기 좋고 활동량 적고 순한 성격이면 좋겠어")]}
    body, update = run_filter({"items": rows, "total": 2}, state)
    ids = [r["desertionNo"] for r in body["items"]]
    assert ids[0] == "A1"
    assert "특이사항" in body["items"][0]["match_reason"]
    assert update["last_search_filters"]["gentle"] is True
    assert update["last_search_filters"]["low_activity"] is True
    assert update["last_search_filters"]["apartment"] is True


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

def test_mixed_off_topic_is_blocked_by_guardrail_classifier(monkeypatch):
    """온토픽 단어가 섞여도 가드레일 분류 결과로 차단한다."""
    from petpal import middleware as mw

    class FakeClassifier:
        def with_structured_output(self, schema):
            return self

        def invoke(self, prompt):
            return GuardrailClassification(label="off_topic", confidence=0.98, blocked=True)

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
