"""설계서 2.3의 미확정 항목 확인 — 모델이 temperature 커스텀 값을 받는지 1회 호출로 검증한다.

    python scripts/probe_temperature.py
결과에 따라 .env 의 PETPAL_TEMPERATURE 를 on/off 로 고정하면 된다.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import dotenv_values, load_dotenv

# override=True: 셸에 남은 옛 키가 .env 를 가리지 않도록 이 프로젝트에서는 .env 를 우선한다.
# 또한 "어느 .env 를 읽었는지" 혼동이 잦아, 이 스크립트는 프로젝트 루트의 .env 만 명시적으로 읽는다.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DOTENV_PATH = _PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=_DOTENV_PATH, override=True)


def _looks_like_placeholder(key: str) -> bool:
    k = (key or "").strip()
    if not k:
        return True
    if set(k) <= {"*"}:
        return True
    upper = k.upper()
    if "YOUR_" in upper or "REPLACE" in upper or "DUMMY" in upper:
        return True
    return False


def _diagnose_auth_env() -> str | None:
    key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not key:
        return (
            "OPENAI_API_KEY 가 비어 있습니다. `cp .env.example .env` 후 `.env`에 키를 넣어주세요."
        )

    # .env 존재 여부/포함 여부만 확인(값은 출력하지 않음)
    try:
        env_values = dotenv_values(_DOTENV_PATH) if _DOTENV_PATH.exists() else {}
        env_has_key = bool((env_values.get("OPENAI_API_KEY") or "").strip())
    except Exception:
        env_has_key = False

    if _looks_like_placeholder(key):
        where = ".env" if env_has_key else "환경변수"
        return f"OPENAI_API_KEY 가 placeholder 처럼 보입니다({where}). 실제 키로 교체해 주세요."

    # 너무 많이 노출하지 않도록 형태만 힌트
    has_quotes = (key[:1] in {"'", '"'} or key[-1:] in {"'", '"'})
    has_space = any(ch.isspace() for ch in key)
    looks_like_openai = key.startswith(("sk-", "sk-proj-"))
    return (
        "OPENAI_API_KEY 감지됨("
        f"len={len(key)}, in_dotenv={env_has_key}, "
        f"looks_like_openai_key={looks_like_openai}, has_quotes={has_quotes}, has_space={has_space})"
    )


def probe(model_name: str, temperature: float = 0.3) -> str:
    from langchain.chat_models import init_chat_model

    try:
        init_chat_model(model_name, temperature=temperature).invoke("ping")
        return "지원함  → PETPAL_TEMPERATURE=on"
    except Exception as exc:
        text = str(exc)
        if "error code: 401" in text.lower() or "incorrect api key" in text.lower():
            hint = _diagnose_auth_env()
            if hint:
                return f"확인 실패(인증 오류): {hint}"
            return "확인 실패(인증 오류): OPENAI_API_KEY 를 확인해 주세요."
        if "temperature" in text.lower():
            return "거부함  → PETPAL_TEMPERATURE=off (reasoning_effort/verbosity 사용 검토)"
        return f"확인 실패(다른 오류): {text[:160]}"


if __name__ == "__main__":
    models = sys.argv[1:] or ["gpt-5-mini", "gpt-5-nano"]
    for name in models:
        print(f"{name:14} {probe(name)}")
