"""환경 설정과 모델 팩토리. 설계서 1.5 기술/보안 항목에 대응한다."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv

# override=True: 셸에 남은 옛 키가 .env 를 가리지 않도록 이 프로젝트에서는 .env 를 우선한다.
load_dotenv(override=True)

log = logging.getLogger("petpal")

# 설계서 1.5 기타 — 두 공공 API의 Base URL
ANIMAL_BASE = "http://apis.data.go.kr/1543061/abandonmentPublicService_v2"
TRAVEL_BASE = "https://apis.data.go.kr/B551011/KorPetTourService2"

# 구조동물 축종 코드 (upkind)
UPKIND = {"개": "417000", "고양이": "422400", "기타": "429900"}
UPKIND_NAME = {v: k for k, v in UPKIND.items()}

# 동반여행 카테고리 (contentTypeId)
CONTENT_TYPE = {
    "관광지": "12",
    "문화시설": "14",
    "레포츠": "28",
    "숙박": "32",
    "쇼핑": "38",
    "음식점": "39",
}
CONTENT_TYPE_NAME = {v: k for k, v in CONTENT_TYPE.items()}


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


@dataclass(frozen=True)
class Settings:
    """설계서 1.5 보안: 인증키는 환경변수로만 관리하고 코드에 두지 않는다.

    data.go.kr 는 계정당 인증키 하나를 발급하며, 승인된 모든 API 에 공용으로 쓴다.
    구조동물·동반여행 두 서비스 모두 이 키 하나로 호출한다.
    """

    service_key: str
    orchestrator_model: str = "gpt-5-mini"
    classifier_model: str = "gpt-5-mini"
    orchestrator_temperature: float = 0.3
    classifier_temperature: float = 0.0
    http_timeout: float = 8.0
    max_retries: int = 3            # 1.5 안정성 — 3회 재시도
    backoff_factor: float = 2.0
    run_limit: int = 6              # 2.3 max iterations
    animal_cache_ttl: int = 300     # 1.5 성능 — 공고는 변동이 잦아 짧은 TTL
    travel_cache_ttl: int = 86_400  # 여행지는 변동이 적어 긴 TTL
    detail_fanout: int = 5          # 2.2 6단계 — detailPetTour2 병렬 호출 상한
    default_search_days: int = 30   # bgnde/endde 기본 조회 기간

    @classmethod
    def load(cls) -> "Settings":
        key = _env("DATA_GO_KR_SERVICE_KEY")
        if not key:
            raise RuntimeError(
                "DATA_GO_KR_SERVICE_KEY 가 없습니다. .env.example 을 .env 로 복사한 뒤 키를 채워 주세요."
            )
        return cls(service_key=key)


def temperature_supported(model_name: str) -> bool:
    """GPT-5 계열은 temperature 커스텀 값을 거부할 수 있어 기본적으로 생략한다.

    설계서 2.3의 미확정 항목. `python -m scripts.probe_temperature` 로 실제 지원 여부를
    한 번 확인한 뒤 PETPAL_TEMPERATURE=on/off 로 고정하는 것을 권장한다.
    """
    flag = _env("PETPAL_TEMPERATURE", "auto").lower()
    if flag in {"on", "true", "1"}:
        return True
    if flag in {"off", "false", "0"}:
        return False
    return not model_name.startswith(("gpt-5", "o1", "o3", "o4"))


def build_model(model_name: str, temperature: float):
    """init_chat_model 래퍼. temperature 미지원 모델이면 인자를 빼고 만든다."""
    from langchain.chat_models import init_chat_model

    if temperature_supported(model_name):
        return init_chat_model(model_name, temperature=temperature)
    log.debug("%s: temperature 인자를 생략합니다(미지원 가능성).", model_name)
    return init_chat_model(model_name)
