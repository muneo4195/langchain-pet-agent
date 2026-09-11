"""대화형 실행기.  사용법:  python -m petpal.cli  [--user U] [--thread T]"""

from __future__ import annotations

import argparse
import logging
import sys
import uuid

from .agent import build_agent
from .context import PetPalContext
from .middleware import text_of
from .schemas import AgentResponse

BANNER = """
  함께갈개 — 유기동물 입양 매칭 & 반려동물 동반여행 컨시어지
  종료: /quit    상태 초기화: /reset
"""


def render(state: dict) -> str:
    out: list[str] = []
    guardrail = state.get("guardrail") or {}
    if guardrail.get("blocked"):
        messages = state.get("messages") or []
        return text_of(messages[-1]) if messages else "요청이 차단되었습니다."

    response = state.get("structured_response")
    if isinstance(response, AgentResponse):
        out.append(response.message)
        for card in response.animals:
            urgency = f"  [마감임박:{card.urgency}]" if card.urgency == "high" else ""
            out.append(
                f"  · {card.kind_name} ({card.desertion_no}) — {card.shelter_name}"
                f"  적합도 {card.match_score:.2f}{urgency}\n    {card.match_reason}"
            )
        for card in response.places:
            extra = f" / {card.allowed_species}" if card.allowed_species else ""
            out.append(f"  · {card.place_name} ({card.content_id}) — {card.acmpy_type}{extra}")
            if card.caution:
                out.append(f"    유의: {card.caution}")
        if response.no_result_hint:
            out.append(f"  → {response.no_result_hint}")
        out.append(f"  [grounded={response.grounded}]")
    else:
        messages = state.get("messages") or []
        if messages:
            last = messages[-1]
            out.append(text_of(last))
    return "\n".join(out) or "(응답 없음)"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="함께갈개 AI Agent")
    parser.add_argument("--user", default="demo-user", help="Runtime Context 의 user_id")
    parser.add_argument("--thread", default=None, help="대화 스레드 ID(생략 시 새로 생성)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("question", nargs="*", help="한 번만 물어보고 종료")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    agent = build_agent()
    context = PetPalContext(user_id=args.user)
    thread_id = args.thread or str(uuid.uuid4())

    def ask(text: str) -> None:
        config = {"configurable": {"thread_id": thread_id}}
        state = agent.invoke({"messages": [{"role": "user", "content": text}]},
                            config=config, context=context)
        print(render(state), "\n")

    if args.question:
        ask(" ".join(args.question))
        return 0

    print(BANNER)
    while True:
        try:
            text = input("나 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not text:
            continue
        if text in {"/quit", "/exit"}:
            return 0
        if text == "/reset":
            thread_id = str(uuid.uuid4())
            print("(새 대화를 시작합니다)\n")
            continue
        try:
            ask(text)
        except Exception as exc:  # 대화가 통째로 끊기지 않도록
            print(f"오류: {exc}\n")


if __name__ == "__main__":
    sys.exit(main())
