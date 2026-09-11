"""설계서 3.1 Context — State 스키마."""

from __future__ import annotations

from typing import Any, NotRequired

from langchain.agents.middleware import AgentState

from .schemas import AgentResponse


class PetPalState(AgentState):
    """대화 중 변하는 값. Runtime Context(user_id)·Store(user_preference)와 구분된다."""

    intent: NotRequired[str]                      # before_agent 가 산출한 의도 라벨
    guardrail: NotRequired[dict[str, Any]]        # 가드레일 판정 로그(사용자 비노출)
    selected_animal: NotRequired[dict[str, Any]]  # 연계 추천의 기준이 되는 동물
    last_search_filters: NotRequired[dict[str, Any]]
    last_tool_results: NotRequired[dict[str, Any]]  # grounded 대조 근거
    structured_response: NotRequired[AgentResponse | None]
