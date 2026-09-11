"""설계서 2.3의 미확정 항목 확인 — 모델이 temperature 커스텀 값을 받는지 1회 호출로 검증한다.

    python scripts/probe_temperature.py
결과에 따라 .env 의 PETPAL_TEMPERATURE 를 on/off 로 고정하면 된다.
"""

from __future__ import annotations

import sys

from dotenv import load_dotenv

load_dotenv()


def probe(model_name: str, temperature: float = 0.3) -> str:
    from langchain.chat_models import init_chat_model

    try:
        init_chat_model(model_name, temperature=temperature).invoke("ping")
        return "지원함  → PETPAL_TEMPERATURE=on"
    except Exception as exc:
        text = str(exc)
        if "temperature" in text.lower():
            return "거부함  → PETPAL_TEMPERATURE=off (reasoning_effort/verbosity 사용 검토)"
        return f"확인 실패(다른 오류): {text[:160]}"


if __name__ == "__main__":
    models = sys.argv[1:] or ["gpt-5-mini", "gpt-5-nano"]
    for name in models:
        print(f"{name:14} {probe(name)}")
