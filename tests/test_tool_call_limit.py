"""S-08 설계서 테스트 시나리오 — max iteration 도달 시 동작.

petpal.agent 가 실제로 조립하는 값(설계서 2.3 max iterations = run_limit,
exit_behavior="continue")으로 langchain 내장 ToolCallLimitMiddleware 가 어떻게
동작하는지 LLM 호출 없이 직접 검증한다.
"""

from langchain.agents.middleware.tool_call_limit import ToolCallLimitMiddleware
from langchain.messages import AIMessage

from petpal.config import Settings

RUN_LIMIT = Settings(service_key="test-key").run_limit


def ai_message_with_tool_calls(n: int) -> AIMessage:
    return AIMessage(content="", tool_calls=[
        {"name": "search_rescued_animals", "args": {}, "id": f"call-{i}", "type": "tool_call"}
        for i in range(n)
    ])


def test_run_limit_matches_design_max_iterations():
    assert RUN_LIMIT == 6, "설계서 2.3 max iterations 값이 바뀌면 이 테스트도 함께 갱신한다"


def test_calls_within_limit_pass_through_untouched():
    mw = ToolCallLimitMiddleware(run_limit=RUN_LIMIT, exit_behavior="continue")
    state = {"messages": [ai_message_with_tool_calls(RUN_LIMIT)]}
    update = mw.after_model(state, None)

    assert update is not None
    assert update["run_tool_call_count"]["__all__"] == RUN_LIMIT
    assert "messages" not in update, "한도 안이면 차단 메시지를 추가하지 않는다"


def test_exceeding_run_limit_blocks_only_the_extra_calls():
    """petpal 설정(exit_behavior='continue')은 초과분만 막고 대화 자체를 끝내지 않는다."""
    mw = ToolCallLimitMiddleware(run_limit=RUN_LIMIT, exit_behavior="continue")
    state = {"messages": [ai_message_with_tool_calls(RUN_LIMIT + 2)]}
    update = mw.after_model(state, None)

    assert "jump_to" not in (update or {}), "continue 모드는 대화를 강제 종료하지 않는다(TS-08 무한루프 방지와 별개)"
    blocked = update["messages"]
    assert len(blocked) == 2, "한도를 넘긴 2건에만 에러 메시지가 붙어야 한다"
    assert all(m.status == "error" for m in blocked)
    assert all("call limit exceeded" in m.content.lower() for m in blocked)
    assert update["run_tool_call_count"]["__all__"] == RUN_LIMIT + 2


def test_second_batch_in_same_run_is_blocked_immediately_once_limit_reached():
    """한 런 안에서 이미 한도에 도달했으면 다음 배치는 전부 차단된다(무한 반복 방지)."""
    mw = ToolCallLimitMiddleware(run_limit=RUN_LIMIT, exit_behavior="continue")

    state = {"messages": [ai_message_with_tool_calls(RUN_LIMIT)]}
    first = mw.after_model(state, None)

    state = {
        "messages": [ai_message_with_tool_calls(1)],
        "run_tool_call_count": first["run_tool_call_count"],
        "thread_tool_call_count": first["thread_tool_call_count"],
    }
    second = mw.after_model(state, None)

    assert len(second["messages"]) == 1
    assert second["messages"][0].status == "error"
