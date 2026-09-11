"""설계서 2.3 / 3.2 — Agent 조립."""

from __future__ import annotations

from langchain.agents import create_agent
from langchain.agents.middleware import (
    SummarizationMiddleware,
    ToolCallLimitMiddleware,
    ToolRetryMiddleware,
)
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from .config import Settings, build_model
from .context import PetPalContext
from .middleware import CUSTOM_MIDDLEWARE
from .schemas import AgentResponse
from .state import PetPalState
from .tools import ALL_TOOLS

# 설계서 2.3 System Prompt
SYSTEM_PROMPT = """당신은 유기동물 입양 상담과 반려동물 동반여행 추천을 돕는 '함께갈개'입니다.

반드시 Tool 호출 결과에 근거해서만 동물·여행지의 구체적인 이름/위치/연락처를 언급하고,
확인되지 않은 정보는 추측하지 않고 '미확인'으로 안내합니다.
검색 결과가 0건이면 결과를 지어내지 말고, 조건을 넓히는 대안(인접 시군구·다른 카테고리)을 제안합니다.
안락사·자연사 등 민감한 상태 정보는 담담하고 정중한 어조로 전달합니다.
적합도 점수는 시스템이 계산해 전달한 값(match_score)만 그대로 인용하고 직접 매기지 않습니다.

작업 규칙
- 지역이나 조건이 주어지면 되묻지 말고 먼저 Tool 로 검색합니다. 지역명은 자연어 그대로
  넘기면 시스템이 코드로 변환합니다. 결과를 보여준 뒤에 필요한 추가 조건을 제안하세요.
- 동반 조건(동반유형·동반가능동물·필요사항)은 목록 조회에 포함되지 않습니다.
  필요하면 search_pet_friendly_travel 결과의 content_id 상위 3~5건으로 get_pet_travel_detail 을 호출하세요.
- 특정 동물의 상세를 물으면 직전 검색 결과의 desertionNo 로 get_animal_detail 을 호출하세요.
- 최종 답변은 AgentResponse 스키마로 정리하되, animals/places 에는 Tool 결과에 실제로 있던 항목만 담습니다.
- grounded 필드는 시스템이 검증해 채우므로 임의로 true 로 두지 마세요.
"""


def build_agent(*, store=None, checkpointer=None, settings: Settings | None = None):
    """설계서 3.2 표 그대로의 미들웨어 구성으로 에이전트를 만든다."""
    s = settings or Settings.load()
    model = build_model(s.orchestrator_model, s.orchestrator_temperature)

    middleware = [
        *CUSTOM_MIDDLEWARE,
        # 내장 미들웨어 — 3.2 표의 Built-in 행
        ToolRetryMiddleware(max_retries=s.max_retries, backoff_factor=s.backoff_factor),
        ToolCallLimitMiddleware(run_limit=s.run_limit, exit_behavior="continue"),
        SummarizationMiddleware(model=build_model(s.classifier_model, 0.0), trigger=("messages", 40)),
    ]

    return create_agent(
        model=model,
        tools=ALL_TOOLS,
        system_prompt=SYSTEM_PROMPT,
        middleware=middleware,
        response_format=AgentResponse,
        state_schema=PetPalState,
        context_schema=PetPalContext,
        store=store if store is not None else InMemoryStore(),
        checkpointer=checkpointer if checkpointer is not None else InMemorySaver(),
        name="함께갈개",
    )
