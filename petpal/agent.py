"""설계서 2.3 / 3.2 — Agent 조립."""

from __future__ import annotations

from langchain.agents import create_agent
from langchain.agents.middleware import (
    SummarizationMiddleware,
    ToolCallLimitMiddleware,
    ToolRetryMiddleware,
)
from .config import Settings, build_model
from .context import PetPalContext
from .middleware import CUSTOM_MIDDLEWARE
from .persistence import build_persistence
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
- 되묻기 전에 반드시 먼저 검색합니다. 지역이 하나라도 언급됐으면 그대로 Tool 에 넘기세요.
  지역명은 자연어 그대로 넘기면 시스템이 코드로 변환합니다. 조건이 부족해도 일단 검색해
  결과를 보여준 다음, 그 뒤에 좁힐 조건을 제안하세요. 검색 한 번도 하지 않고 선택지만
  나열하는 답변은 하지 않습니다.
- 동반 조건(동반유형·동반가능동물·필요사항)은 search_pet_friendly_travel 결과에 시스템이
  자동으로 붙여 줍니다. 결과의 acmpy_type / allowed_species / caution 을 그대로 쓰세요.
  특정 장소를 다시 확인해야 할 때만 get_pet_travel_detail 을 호출합니다.
- 검색 결과가 0건이면 recent_days 를 넓히거나(예: 30 → 90) 인접 지역·다른 카테고리로
  한 번 더 시도한 뒤에 사용자에게 알립니다.
- 입양 동물의 판매·전매·영리 목적 양도는 돕지 않습니다. 판매 계약서 초안, 가격 책정,
  구매자 심사표 같은 것도 작성하지 않습니다.
- 더 이상 돌보기 어렵다는 상담(파양)에는 짧게만 답합니다: 원래 입양처나 관할 보호소에
  먼저 연락하도록 안내하고, 절대 유기하지 말 것을 알립니다. 재입양 알선 절차를
  단계별로 코칭하지는 않습니다.
- 특정 동물의 상세를 물으면 직전 검색 결과의 desertionNo 로 get_animal_detail 을 호출하세요.
- 최종 답변은 AgentResponse 스키마로 정리하되, animals/places 에는 Tool 결과에 실제로 있던 항목만 담습니다.
- grounded 필드는 시스템이 검증해 채우므로 임의로 true 로 두지 마세요.
- 성격(순함)·활동량·아파트/실내 적합 같은 특성은 공고의 특이사항(specialMark)에 명시적으로 언급된 경우에만 근거로 사용합니다.
  특이사항에 언급이 없으면 '미확인'이라고 말하고 추측하지 않습니다.
"""


def build_agent(*, store=None, checkpointer=None, settings: Settings | None = None,
                in_memory: bool = False):
    """설계서 3.2 표 그대로의 미들웨어 구성으로 에이전트를 만든다."""
    s = settings or Settings.load()
    model = build_model(s.orchestrator_model, s.orchestrator_temperature)
    if checkpointer is None or store is None:
        default_checkpointer, default_store = build_persistence(in_memory=in_memory)
        checkpointer = checkpointer if checkpointer is not None else default_checkpointer
        store = store if store is not None else default_store

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
        store=store,
        checkpointer=checkpointer,
        name="함께갈개",
    )
