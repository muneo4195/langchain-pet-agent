"""설계서 3.3 '근거 없는 사실 차단' — 모델의 자기 신고를 믿지 않는다는 것이 핵심."""

from langchain.messages import AIMessage

from petpal.middleware import output_guardrail
from petpal.schemas import AgentResponse, AnimalCard, PetTravelCard


def card(no="A1"):
    return AnimalCard(desertion_no=no, kind_name="푸들", shelter_name="서울보호소",
                      match_score=1.0, match_reason="조건 일치")


def run(response, results):
    return output_guardrail.after_agent(
        {"messages": [], "structured_response": response, "last_tool_results": results}, None)


def test_hallucinated_card_is_dropped():
    """Tool 응답에 없던 공고번호를 모델이 지어내면 제거하고 grounded=False 로 내린다."""
    response = AgentResponse(response_type="animal_list", message="두 마리 찾았어요",
                             animals=[card("A1"), card("A9")], grounded=True)
    fixed = run(response, {"animals": {"A1": {}}})["structured_response"]
    assert [c.desertion_no for c in fixed.animals] == ["A1"]
    assert fixed.grounded is False


def test_model_self_report_is_overwritten():
    """모델이 grounded=True 라고 해도 시스템이 대조해 다시 판정한다."""
    response = AgentResponse(response_type="animal_list", message="한 마리", animals=[card("A1")], grounded=False)
    fixed = run(response, {"animals": {"A1": {}}})["structured_response"]
    assert fixed.grounded is True


def test_all_dropped_falls_back_to_safe_message():
    response = AgentResponse(response_type="animal_list", message="세 마리 찾았어요!",
                             animals=[card("Z1")], grounded=True)
    fixed = run(response, {"animals": {"A1": {}}})["structured_response"]
    assert fixed.animals == [] and "확인된 정보가 없습니다" in fixed.message


def test_places_are_checked_too():
    response = AgentResponse(
        response_type="travel_list", message="숙소 안내",
        places=[PetTravelCard(content_id="C1", place_name="고심스테이"),
                PetTravelCard(content_id="C9", place_name="없는숙소")])
    fixed = run(response, {"places": {"C1": {}}})["structured_response"]
    assert [c.content_id for c in fixed.places] == ["C1"]


def test_response_type_is_overridden_by_intent_mapping():
    """response_type 은 모델이 임의로 정하지 않고 intent/내용으로 시스템이 보정한다."""
    response = AgentResponse(
        response_type="general_chat",
        message="두 마리 찾았어요",
        animals=[card("A1")],
    )
    out = output_guardrail.after_agent(
        {"messages": [], "intent": "adoption_search", "structured_response": response, "last_tool_results": {"animals": {"A1": {}}}},
        None,
    )
    fixed = out["structured_response"]
    assert fixed.response_type == "animal_list"


def test_response_type_prefers_places_when_only_places_present():
    response = AgentResponse(
        response_type="animal_list",
        message="장소 안내",
        places=[PetTravelCard(content_id="C1", place_name="고심스테이")],
    )
    out = output_guardrail.after_agent(
        {"messages": [], "intent": "travel_search", "structured_response": response, "last_tool_results": {"places": {"C1": {}}}},
        None,
    )
    fixed = out["structured_response"]
    assert fixed.response_type == "travel_list"


def test_response_type_falls_back_to_intent_when_no_cards():
    response = AgentResponse(
        response_type="animal_list",
        message="대화",
        animals=[],
        places=[],
    )
    out = output_guardrail.after_agent(
        {"messages": [], "intent": "general_chat", "structured_response": response, "last_tool_results": {}},
        None,
    )
    fixed = out["structured_response"]
    assert fixed.response_type == "general_chat"


def test_response_type_prefers_animals_when_both_present():
    """비정상 출력(animals+places 동시)일 때도 시스템 보정이 결정적이어야 한다."""
    response = AgentResponse(
        response_type="general_chat",
        message="둘 다",
        animals=[card("A1")],
        places=[PetTravelCard(content_id="C1", place_name="고심스테이")],
    )
    out = output_guardrail.after_agent(
        {
            "messages": [],
            "intent": "travel_search",
            "structured_response": response,
            "last_tool_results": {"animals": {"A1": {}}, "places": {"C1": {}}},
        },
        None,
    )
    fixed = out["structured_response"]
    assert fixed.response_type == "animal_list"


def test_pii_masked_in_final_message():
    response = AgentResponse(response_type="general_chat", message="담당자01012345678로 연락하세요. 메일me@example.com입니다")
    fixed = run(response, {})["structured_response"]
    assert "01012345678" not in fixed.message
    assert "me@example.com" not in fixed.message
    assert "010-****-****" in fixed.message
    assert "***@***" in fixed.message


def test_plain_message_path_is_masked():
    """구조화 응답이 없을 때(가드레일 차단 등)도 최종 메시지를 마스킹한다."""
    out = output_guardrail.after_agent(
        {"messages": [AIMessage(content="연락처 010-9999-8888")], "last_tool_results": {}}, None)
    assert "010-9999-8888" not in out["messages"][0].content


def test_pii_masked_inside_cards():
    """message 뿐 아니라 카드 본문의 연락처도 가린다."""
    animal = card("A1").model_copy(update={"match_reason": "담당자 010-1111-2222 문의"})
    place = PetTravelCard(content_id="C1", place_name="가", caution="예약 010-3333-4444")
    response = AgentResponse(response_type="animal_list", message="안내",
                             animals=[animal], places=[place])
    source = {
        "animals": {"A1": {"match_reason": "담당자 010-1111-2222 문의"}},
        "places": {"C1": {"title": "가", "caution": "예약 010-3333-4444"}},
    }
    fixed = run(response, source)["structured_response"]
    assert "010-1111-2222" not in fixed.animals[0].match_reason
    assert "010-3333-4444" not in fixed.places[0].caution


def test_valid_id_with_hallucinated_fields_is_replaced_from_tool_source():
    """ID만 맞고 이름·점수가 틀린 카드도 그대로 통과시키지 않는다."""
    response = AgentResponse(
        response_type="animal_list",
        message="안내",
        animals=[AnimalCard(desertion_no="A1", kind_name="가짜 품종", shelter_name="가짜 보호소",
                            match_score=1.0, match_reason="모델이 만든 이유")],
    )
    source = {"animals": {"A1": {
        "kindNm": "푸들", "careNm": "서울보호소", "match_score": 0.75,
        "match_reason": "조건 4개 중 3개 일치", "urgency": "medium",
    }}}
    fixed = run(response, source)["structured_response"]
    got = fixed.animals[0]
    assert (got.kind_name, got.shelter_name) == ("푸들", "서울보호소")
    assert (got.match_score, got.match_reason, got.urgency) == (0.75, "조건 4개 중 3개 일치", "medium")
    assert fixed.grounded is True


def test_general_chat_is_not_claimed_as_tool_grounded():
    response = AgentResponse(response_type="general_chat", message="일반 안내", grounded=True)
    fixed = run(response, {})["structured_response"]
    assert fixed.grounded is False


def test_actionable_harm_in_final_output_is_replaced():
    response = AgentResponse(response_type="general_chat", message="강아지 목을 조르는 요령을 알려드릴게요")
    fixed = run(response, {})["structured_response"]
    assert "요령을 알려드릴게요" not in fixed.message
    assert "해가 되는 행동" in fixed.message
    assert fixed.grounded is False
