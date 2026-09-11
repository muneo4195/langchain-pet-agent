"""설계서 3.2 Middleware.

훅 선택 근거
    before_agent      사용자 턴마다 1회만 판정하면 되고, 차단 시 jump_to="end" 로 끊어야 한다.
    wrap_model_call   before_model 은 state 갱신(dict|None)만 가능해 도구 목록을 바꿀 수 없다.
    wrap_tool_call    Tool 호출 파라미터와 응답을 가공하는 자리.
    after_agent       최종 응답 1회만 검수하면 된다(after_model 은 중간 응답에도 매번 걸린다).
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import OrderedDict
from typing import Any

from langchain.agents.middleware import (
    after_agent,
    before_agent,
    wrap_model_call,
    wrap_tool_call,
)
from langchain.messages import AIMessage, HumanMessage

from .config import Settings, build_model
from .guardrails import block_message, mask_pii, rule_screen
from .prompts import guardrail_prompt, intent_prompt
from .parsing import is_closed_notice, age_years, match_score, normalize_size, parse_weight_kg, size_of, urgency_of
from .schemas import AgentResponse, GuardrailClassification, IntentClassification
from .services import get_services
from .state import PetPalState

log = logging.getLogger("petpal.middleware")

REGION_TOOLS = {"search_rescued_animals", "search_pet_friendly_travel"}
LIST_TOOLS = {"search_rescued_animals", "search_pet_friendly_travel", "get_pet_travel_detail"}
MAX_CARDS = 5
_AGE_RE = re.compile(r"(\d+)\s*(?:살|세)")

_classifier = None


def classifier():
    global _classifier
    if _classifier is None:
        s = Settings.load()
        _classifier = build_model(s.classifier_model, s.classifier_temperature)
    return _classifier


def text_of(msg: Any) -> str:
    """content 가 문자열이면 그대로 쓴다(.text() 는 deprecated)."""
    content = getattr(msg, "content", "")
    if isinstance(content, str):
        return content
    text = getattr(msg, "text", None)
    return text if isinstance(text, str) else str(content)


def _last_human(messages: list[Any]) -> str:
    for msg in reversed(messages or []):
        if isinstance(msg, HumanMessage):
            return text_of(msg)
    return ""


# ────────────────────────────────────────────── ① 입력 가드레일 + 의도 분류
@before_agent(state_schema=PetPalState, can_jump_to=["end"])
def input_guardrail(state: PetPalState, runtime) -> dict[str, Any] | None:
    """규칙 필터 1차 → GPT-5-nano 2차. 위반이면 Agent 호출 자체를 중단한다."""
    text = _last_human(state["messages"])
    if not text:
        return None

    previous_intent = state.get("intent")
    label, decided = rule_screen(text)
    confidence = 1.0 if decided else 0.0

    if not decided:  # 애매한 것만 모델에게 넘긴다(설계 원칙: 저비용 필터 우선)
        try:
            verdict = classifier().with_structured_output(GuardrailClassification).invoke(
                guardrail_prompt(text, previous_intent)
            )
            label, confidence = verdict.label, verdict.confidence
        except Exception as exc:
            # fail-open 금지 — 규칙 필터가 이미 off_topic 후보로 본 입력이므로 그 판정을 유지한다.
            log.warning("가드레일 분류 실패, 규칙 판정(%s)을 유지합니다: %s", label, exc)
            confidence = 0.0

    if label != "normal":
        return {
            "messages": [AIMessage(content=block_message(label, text))],
            "jump_to": "end",
            "guardrail": {"label": label, "confidence": confidence, "blocked": True},
        }

    # 지시어("거기", "그 아이")는 직전 의도를 모르면 판단할 수 없어 힌트로 넘긴다.
    intent = "general_chat"
    intent_confidence = 0.0
    try:
        intent_verdict = classifier().with_structured_output(IntentClassification).invoke(
            intent_prompt(text, previous_intent)
        )
        intent = intent_verdict.intent
        intent_confidence = intent_verdict.confidence
    except Exception as exc:
        log.warning("의도 분류 실패, 전체 Tool 노출: %s", exc)
        intent = ""

    # 정상 키워드를 끼운 도메인 외 질문은 규칙만으로 열거하기 어렵다. 이미 수행한
    # IntentClassification 한 번의 결과를 재사용해 추가 모델 호출 없이 차단한다.
    if intent == "off_topic":
        return {
            "messages": [AIMessage(content=block_message("off_topic", text))],
            "jump_to": "end",
            "guardrail": {
                "label": "off_topic", "confidence": intent_confidence, "blocked": True,
            },
        }

    return {"intent": intent, "guardrail": {"label": "normal", "confidence": confidence, "blocked": False}}


# ────────────────────────────────────────────── ② 의도별 Tool 가시성 제한
@wrap_model_call(state_schema=PetPalState)
def intent_routing(request, handler):
    """before_model 로는 도구 목록을 바꿀 수 없어 이 훅에서 request.override(tools=...) 한다."""
    intent = (request.state or {}).get("intent") or ""
    if not intent:
        return handler(request)  # 라벨이 없으면 전체 노출(안전한 기본값)

    if intent.startswith("adoption"):
        keep = {"search_rescued_animals", "get_animal_detail", "save_user_preference"}
    elif intent.startswith("travel"):
        keep = {"search_pet_friendly_travel", "get_pet_travel_detail", "save_user_preference"}
    else:
        # general_chat 으로 잘못 분류되더라도 검색 능력을 잃지 않도록 제한하지 않는다.
        return handler(request)

    # response_format 이 붙인 구조화 출력용 도구는 이름이 우리 Tool 목록에 없으므로 항상 남긴다.
    ours = {"search_rescued_animals", "get_animal_detail", "search_pet_friendly_travel",
            "get_pet_travel_detail", "save_user_preference"}
    tools = [t for t in request.tools
             if getattr(t, "name", None) in keep or getattr(t, "name", None) not in ours]
    return handler(request.override(tools=tools) if tools else request)


# ────────────────────────────────────────────── ③ 선호 정보 프롬프트 주입
@wrap_model_call(state_schema=PetPalState)
def preference_binding(request, handler):
    """Store 의 user_preference 를 System Prompt 에 덧붙인다."""
    store = getattr(request.runtime, "store", None)
    user_id = getattr(getattr(request.runtime, "context", None), "user_id", None)
    if not store or not user_id:
        return handler(request)
    try:
        item = store.get(("preferences",), user_id)
    except Exception as exc:
        log.warning("선호 정보 조회 실패: %s", exc)
        return handler(request)
    if not item or not item.value:
        return handler(request)

    base = request.system_message.content if request.system_message else ""
    pref = ", ".join(f"{k}={v}" for k, v in item.value.items())
    return handler(request.override(system_prompt=f"{base}\n\n[기억된 사용자 선호] {pref}"))


# ────────────────────────────────────────────── ④ 지역명 → 지역코드 변환·주입
@wrap_tool_call(state_schema=PetPalState)
def region_code_resolver(request, handler):
    """지역이 비어 있는 지시어성 요청("그 아이 사는 곳 근처")을 선택된 동물의 지역으로 채운다.

    지역명 → API 코드 변환 자체는 Tool 이 앱 레벨 캐시를 조회해 수행한다.
    ToolRuntime 이 미들웨어 체인보다 먼저 만들어지고 LangGraph 가 InjectedToolArg 키를
    args 에서 제거하기 때문에, 여기서 코드를 주입해도 Tool 까지 전달되지 않는다.
    """
    call = request.tool_call
    if call["name"] not in REGION_TOOLS:
        return handler(request)

    args = dict(call.get("args") or {})
    if (args.get("region") or "").strip():
        return handler(request)

    selected = (request.state or {}).get("selected_animal") or {}
    fallback = selected.get("orgNm") or selected.get("careAddr") or ""
    if not fallback:
        return handler(request)

    args["region"] = fallback
    log.debug("지역이 비어 selected_animal 의 지역으로 채웁니다: %s", fallback)
    return handler(request.override(tool_call={**call, "args": args}))


# ────────────────────────────────────────────── ⑤ Tool 응답 캐싱 (TTL)
# 프로세스가 오래 살아도 무한히 커지지 않도록 LRU 상한을 둔다.
_CACHE_MAX_ENTRIES = 256
_cache: OrderedDict[str, tuple[float, Any]] = OrderedDict()


@wrap_tool_call(state_schema=PetPalState)
def tool_cache(request, handler):
    """일일 트래픽 1,000건 제한이 있어 캐싱은 최적화가 아니라 필수 제약이다."""
    call = request.tool_call
    if call["name"] not in LIST_TOOLS:
        return handler(request)

    s = get_services().settings
    ttl = s.animal_cache_ttl if call["name"] == "search_rescued_animals" else s.travel_cache_ttl
    key = json.dumps([call["name"], call.get("args")], sort_keys=True, ensure_ascii=False, default=str)

    hit = _cache.get(key)
    if hit and time.time() - hit[0] < ttl:
        _cache.move_to_end(key)
        log.debug("cache hit: %s", call["name"])
        return hit[1].model_copy(update={"tool_call_id": call["id"]})
    _cache.pop(key, None)  # 만료분 제거

    result = handler(request)
    if getattr(result, "content", None) and "api_failed" not in str(result.content):
        _cache[key] = (time.time(), result)
        _cache.move_to_end(key)
        while len(_cache) > _CACHE_MAX_ENTRIES:
            _cache.popitem(last=False)
    return result


# ────────────────────────────────────────────── ⑥ 결과 1차 필터링
@wrap_tool_call(state_schema=PetPalState)
def result_filter(request, handler):
    """종결 공고·폐업 시설·조건 미달 항목을 Tool 응답에서 제거하고 근거를 State 에 기록한다."""
    from langgraph.types import Command

    result = handler(request)
    name = request.tool_call["name"]
    if name not in {"search_rescued_animals", "search_pet_friendly_travel",
                    "get_pet_travel_detail", "get_animal_detail"}:
        return result

    try:
        payload = json.loads(result.content)
    except (TypeError, ValueError):
        return result
    if not isinstance(payload, dict) or payload.get("error"):
        return result

    state = request.state or {}
    prev = dict(state.get("last_tool_results") or {})
    update: dict[str, Any] = {}

    if name == "search_rescued_animals":
        wanted = _wanted_from(state, request.tool_call.get("args") or {})
        kept = []
        for row in payload.get("items", []):
            if is_closed_notice(row.get("processState")):
                continue
            if wanted.get("size") and size_of(parse_weight_kg(row.get("weight"))) != wanted["size"]:
                continue
            if wanted.get("max_age") is not None:
                got = age_years(row.get("age"))
                if got is None or got > wanted["max_age"]:
                    continue
            score, reason = match_score(row, wanted)
            row["match_score"], row["match_reason"] = score, reason
            row["urgency"] = urgency_of(row.get("noticeEdt"))
            kept.append(row)
        kept.sort(key=lambda r: r["match_score"], reverse=True)
        kept = kept[:MAX_CARDS]
        payload["items"], payload["total"], payload["filtered_out"] = kept, len(kept), payload.get("total", 0) - len(kept)
        if not kept:
            days = (request.tool_call.get("args") or {}).get("recent_days", 30)
            payload["no_result_hint"] = (
                f"최근 {days}일 공고 중에는 조건에 맞는 항목이 없습니다. "
                f"recent_days 를 {days * 3} 로 넓히거나 인접 시군구·다른 축종으로 다시 검색할 수 있습니다."
            )
        prev["animals"] = {**(prev.get("animals") or {}), **{r["desertionNo"]: r for r in kept}}
        update["last_search_filters"] = wanted

    elif name == "search_pet_friendly_travel":
        rows = payload.get("items", [])[:MAX_CARDS]
        # 2.2 흐름 6단계 — 동반 조건은 목록 API 에 없으므로 여기서 상세를 붙인다.
        # 모델의 판단에 맡기면 건너뛰는 경우가 있어 파이프라인에서 항상 수행한다.
        if rows:
            from .tools import fetch_pet_details

            details = fetch_pet_details([str(r.get("contentid")) for r in rows])
            for row in rows:
                row.update(_normalize_detail(details.get(str(row.get("contentid")), {})))
        payload["items"], payload["total"] = rows, len(rows)
        if not rows:
            payload["no_result_hint"] = "등록된 동반 가능 장소가 없습니다. 인접 시군구나 다른 카테고리로 넓혀볼 수 있습니다."
        prev["places"] = {**(prev.get("places") or {}), **{str(r.get("contentid")): r for r in rows}}

    elif name == "get_pet_travel_detail":
        details = {cid: _normalize_detail(row) for cid, row in (payload.get("items") or {}).items()}
        payload["items"] = details
        places = dict(prev.get("places") or {})
        for cid, det in details.items():
            places.setdefault(cid, {}).update(det)
        prev["places"] = places

    elif name == "get_animal_detail" and payload.get("desertionNo"):
        update["selected_animal"] = payload

    update["last_tool_results"] = prev
    new_msg = result.model_copy(update={"content": json.dumps(payload, ensure_ascii=False)})
    return Command(update={"messages": [new_msg], **update})


def _normalize_detail(raw: dict[str, Any]) -> dict[str, Any]:
    """detailPetTour2 의 빈 문자열을 '미확인'/None 으로 정리한다."""
    return {
        "acmpy_type": (raw.get("acmpyTypeCd") or "").strip() or "미확인",
        "allowed_species": (raw.get("acmpyPsblCpam") or "").strip() or None,
        "caution": " ".join(
            x for x in [(raw.get("acmpyNeedMtr") or "").strip(),
                        (raw.get("relaAcdntRiskMtr") or "").strip()] if x
        )[:150] or None,
    }


def _wanted_from(state: PetPalState, args: dict[str, Any]) -> dict[str, Any]:
    """후처리 조건을 뽑되, 직전 검색 조건과 병합한다.

    "서울 소형견 찾아줘" → "강릉으로 바꿔줘" 처럼 조건 일부만 다시 말하는 경우
    앞서 말한 조건이 사라지면 안 된다(설계서 3.1 last_search_filters 용도).
    이번 발화에서 명시한 값이 이전 값을 덮어쓴다.
    """
    text = _last_human(state.get("messages") or [])
    # 지역은 API 파라미터(upr_cd/org_cd)로 이미 걸러졌으므로 적합도 항목에 넣지 않는다.
    current: dict[str, Any] = {}
    if size := normalize_size(text):
        current["size"] = size
    if m := _AGE_RE.search(text):
        if "이하" in text or "미만" in text or "어린" in text:
            current["max_age"] = int(m.group(1))
    if args.get("kind_name"):
        current["kind_name"] = args["kind_name"]

    previous = dict(state.get("last_search_filters") or {})
    merged = {**previous, **current}
    if merged != current:
        log.debug("이전 검색 조건과 병합: %s + %s → %s", previous, current, merged)
    return merged


# ────────────────────────────────────────────── ⑦ 출력 가드레일
@after_agent(state_schema=PetPalState)
def output_guardrail(state: PetPalState, runtime) -> dict[str, Any] | None:
    """모델의 자기 신고를 믿지 않고, Tool 응답 원문과 대조해 grounded 를 시스템이 판정한다."""
    response: AgentResponse | None = state.get("structured_response")
    results = state.get("last_tool_results") or {}
    known_animals = set((results.get("animals") or {}).keys())
    known_places = set((results.get("places") or {}).keys())

    if not isinstance(response, AgentResponse):
        messages = state.get("messages") or []
        if messages and isinstance(messages[-1], AIMessage):
            masked = mask_pii(text_of(messages[-1]))
            if masked != messages[-1].content:
                return {"messages": [messages[-1].model_copy(update={"content": masked})]}
        return None

    animals = [c for c in response.animals if c.desertion_no in known_animals]
    places = [c for c in response.places if c.content_id in known_places]
    dropped = (len(response.animals) - len(animals)) + (len(response.places) - len(places))
    if dropped:
        log.warning("근거 없는 카드 %d건을 제거했습니다.", dropped)

    animals = [c.model_copy(update={"match_reason": mask_pii(c.match_reason)}) for c in animals]
    places = [
        c.model_copy(update={"caution": mask_pii(c.caution) if c.caution else None})
        for c in places
    ]
    message = mask_pii(response.message)
    if dropped and not (animals or places):
        message = "확인된 정보가 없습니다. 조건을 바꿔 다시 검색해 볼까요?"

    fixed = response.model_copy(update={
        "animals": animals,
        "places": places,
        "message": message,
        "grounded": dropped == 0,
    })
    return {"structured_response": fixed}


# 목록의 앞쪽이 바깥(outer)이다.
# wrap_tool_call 중첩 순서: result_filter → region_code_resolver → tool_cache → (ToolRetry) → Tool
CUSTOM_MIDDLEWARE = [
    input_guardrail,
    intent_routing,
    preference_binding,
    result_filter,
    region_code_resolver,
    tool_cache,
    output_guardrail,
]
