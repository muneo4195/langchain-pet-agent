"""설계서 2.4 Structured Output 설계."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

GuardrailLabel = Literal["off_topic", "injection", "abuse_request", "normal"]
Intent = Literal[
    "adoption_search", "adoption_detail", "travel_search", "travel_detail", "general_chat",
]
ResponseType = Literal["animal_list", "travel_list", "general_chat"]

# intent → response_type 매핑 (설계서 2.4 IntentClassification 설명)
INTENT_TO_RESPONSE: dict[str, ResponseType] = {
    "adoption_search": "animal_list",
    "adoption_detail": "animal_list",
    "travel_search": "travel_list",
    "travel_detail": "travel_list",
    "general_chat": "general_chat",
}


class GuardrailClassification(BaseModel):
    """입력 가드레일 1차 판정 결과."""

    label: GuardrailLabel = Field(description="입력 가드레일 판정 라벨")
    confidence: float = Field(ge=0.0, le=1.0, description="분류 신뢰도")
    blocked: bool = Field(description="즉시 차단 대상 여부")
    reason: str | None = Field(default=None, max_length=50, description="판단 근거(로그용, 비노출)")


class IntentClassification(BaseModel):
    """가드레일 통과 후 2차 의도 분류 결과."""

    intent: Intent = Field(description="Tool 가시성 라우팅에 쓰이는 의도 라벨")
    confidence: float = Field(ge=0.0, le=1.0, description="분류 신뢰도")


class AnimalCard(BaseModel):
    """유기동물 결과 카드."""

    desertion_no: str = Field(description="유기동물 공고번호(desertionNo)")
    kind_name: str = Field(default="", description="품종명(kindNm)")
    shelter_name: str = Field(default="", description="보호소명(careNm)")
    match_score: float = Field(
        ge=0.0, le=1.0, description="시스템이 규칙 기반으로 계산한 적합도(모델이 부여하지 않음)"
    )
    match_reason: str = Field(max_length=100, description="Tool 응답 필드에 근거한 매칭 이유")
    urgency: Literal["low", "medium", "high"] | None = Field(
        default=None, description="공고 마감 임박도(noticeEdt 기준)"
    )


class PetTravelCard(BaseModel):
    """동반여행지 결과 카드."""

    content_id: str = Field(description="한국관광공사 콘텐츠ID(contentid)")
    place_name: str = Field(description="장소명(title)")
    acmpy_type: str = Field(default="미확인", description="동반유형(acmpyTypeCd)")
    allowed_species: str | None = Field(default=None, description="동반가능동물(acmpyPsblCpam)")
    caution: str | None = Field(default=None, max_length=150, description="동반 시 필요사항 요약")


class AgentResponse(BaseModel):
    """최종 응답. animals/places 를 분리해 strict 스키마 호환성을 확보한다."""

    response_type: ResponseType = Field(description="최종 응답의 유형")
    message: str = Field(description="사용자에게 보여줄 최종 안내 문구")
    animals: list[AnimalCard] = Field(default_factory=list, description="response_type=animal_list 결과")
    places: list[PetTravelCard] = Field(default_factory=list, description="response_type=travel_list 결과")
    grounded: bool = Field(
        default=False,
        description="Tool 응답 원문 대조 결과. 모델이 채우지 않으며 OutputGuardrail 이 덮어쓴다.",
    )
    no_result_hint: str | None = Field(
        default=None, description="결과가 0건·소량일 때 제시할 조건 완화 대안"
    )
