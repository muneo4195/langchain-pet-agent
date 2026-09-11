"""함께갈개 — 유기동물 입양 매칭 및 반려동물 동반여행 컨시어지 AI Agent.

설계서 `4조_Agent_설계서_클로드_수정본.docx` 의 구현체.
"""

__all__ = ["build_agent", "PetPalContext", "AgentResponse"]
__version__ = "0.1.0"


def __getattr__(name: str):  # 무거운 임포트를 지연시킨다
    if name == "build_agent":
        from .agent import build_agent

        return build_agent
    if name == "PetPalContext":
        from .context import PetPalContext

        return PetPalContext
    if name == "AgentResponse":
        from .schemas import AgentResponse

        return AgentResponse
    raise AttributeError(name)
