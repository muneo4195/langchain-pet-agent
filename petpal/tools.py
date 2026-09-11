"""설계서 2.5 Tool 설계.

docstring 은 모델이 '언제 이 Tool 을 호출할지' 판단하는 근거이므로 설계서 문장을 그대로 쓴다.
지역명 → API 코드 변환은 앱 레벨 캐시(CodeTables)를 조회해 Tool 안에서 수행한다.
지시어("그 아이 사는 곳") 해석은 RegionCodeResolverMiddleware 가 맡는다.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from typing import Any

from langchain.tools import ToolRuntime, tool

from .api import PublicDataError
from .config import CONTENT_TYPE, UPKIND
from .parsing import urgency_of
from .services import get_services

log = logging.getLogger("petpal.tools")

REGION_MISSING = {
    "error": "region_unresolved",
    "message": "지역을 코드로 변환하지 못했습니다. 시·도나 시·군·구 이름을 알려 주세요.",
    "items": [],
}


def _resolved_region(region: str) -> dict[str, Any]:
    """지역명을 두 API 의 코드로 변환한다.

    변환 규칙 자체는 앱 레벨 캐시(CodeTables)가 단독으로 갖고 있고 여기서는 조회만 한다.
    미들웨어에서 주입하지 않는 이유: ToolRuntime 은 미들웨어 체인보다 먼저 그래프 상태로
    만들어지고, LangGraph 는 모델 위조를 막으려고 InjectedToolArg 키를 args 에서 제거한다.
    따라서 wrap_tool_call 에서는 값을 Tool 까지 전달할 방법이 없다.
    RegionCodeResolverMiddleware 는 대신 지시어("그 아이 사는 곳") 해석을 맡는다.
    """
    m = get_services().codes.resolve(region)
    return {
        "label": m.label,
        "upr_cd": m.upr_cd,
        "org_cd": m.org_cd,
        "l_dong_regn_cd": m.l_dong_regn_cd,
        "l_dong_signgu_cd": m.l_dong_signgu_cd,
    }


def _need_region(resolved: dict[str, Any], *keys: str) -> dict[str, Any] | None:
    if not any(resolved.get(k) for k in keys):
        return REGION_MISSING
    return None


def fetch_pet_details(content_ids: list[str], limit: int | None = None) -> dict[str, dict[str, Any]]:
    """detailPetTour2 를 상위 N건만 병렬 조회한다. Tool 과 미들웨어가 함께 쓴다."""
    svc = get_services()
    ids = [str(c) for c in content_ids][: (limit or svc.settings.detail_fanout)]
    if not ids:
        return {}

    def fetch(cid: str) -> tuple[str, dict[str, Any]]:
        try:
            return cid, svc.client.detail_pet_tour(cid)
        except PublicDataError as exc:
            log.warning("detailPetTour2(%s) 실패: %s", cid, exc)
            return cid, {}

    with ThreadPoolExecutor(max_workers=len(ids)) as pool:
        return dict(pool.map(fetch, ids))


@tool
def search_rescued_animals(
    region: str,
    upkind: str = "",
    kind_name: str = "",
    state: str = "",
    recent_days: int = 30,
    page: int = 1,
) -> dict[str, Any]:
    """지역·축종·품종·공고상태 조건으로 구조(유기)동물 목록을 검색한다. 입양 후보를 찾을 때 사용.

    Args:
        region: 지역명. '서울 마포구', '강릉' 처럼 자연어 그대로 넘긴다.
        upkind: 축종. '개' / '고양이' / '기타' 중 하나. 비우면 전체.
        kind_name: 품종명. 예: '푸들', '한국 고양이'. 비우면 전체.
        state: 'notice'(공고중) 또는 'protect'(보호중). 비우면 전체.
        recent_days: 최근 며칠 이내의 공고를 볼지. 기본 30일. 결과가 없으면 90, 180 으로 넓힌다.
        page: 결과 페이지. 더 보여달라고 하면 2, 3 으로 올린다.

    나이·성별·체중은 이 API 의 검색 파라미터가 아니라 응답 필드다. 크기나 나이 조건은
    이 Tool 을 호출한 뒤 결과에서 걸러지므로 여기서 지정하지 않는다.
    """
    resolved_region = _resolved_region(region)
    if err := _need_region(resolved_region, "upr_cd"):
        return err
    svc = get_services()
    upkind_cd = UPKIND.get(upkind.strip()) if upkind else None
    kind_cd = svc.codes.kind_code(kind_name, upkind_cd) if kind_name else None
    try:
        rows = svc.client.abandonment_public(
            upr_cd=resolved_region.get("upr_cd"),
            org_cd=resolved_region.get("org_cd"),
            upkind=upkind_cd,
            kind=kind_cd,
            state=state or None,
            bgnde=(date.today() - timedelta(days=max(1, recent_days))).strftime("%Y%m%d"),
            num_of_rows=60,
            page_no=max(1, page),
        )
    except PublicDataError as exc:
        log.warning("search_rescued_animals 실패: %s", exc)
        return {"error": "api_failed", "message": "현재 유기동물 공고 조회가 어렵습니다. 잠시 후 다시 시도해 주세요.", "items": []}

    for row in rows:
        row["urgency"] = urgency_of(row.get("noticeEdt"))
    return {
        "items": rows,
        "region": resolved_region.get("label") or region,
        "total": len(rows),
        "recent_days": recent_days,
        "page": page,
    }


@tool
def get_animal_detail(desertion_no: str, runtime: ToolRuntime) -> dict[str, Any]:
    """특정 공고번호(desertionNo)의 구조동물 상세정보와 소속 보호소 연락처를 조회한다.

    Args:
        desertion_no: 직전 검색 결과에 있던 공고번호.
    """
    cached = (runtime.state or {}).get("last_tool_results", {}).get("animals", {})
    row = cached.get(str(desertion_no))
    if not row:
        return {
            "error": "not_found",
            "message": f"공고번호 {desertion_no} 를 직전 검색 결과에서 찾지 못했습니다. 먼저 조건으로 검색해 주세요.",
        }
    return {
        "desertionNo": row.get("desertionNo"),
        "kindNm": row.get("kindNm"),
        "age": row.get("age"),
        "weight": row.get("weight"),
        "sexCd": row.get("sexCd"),
        "neuterYn": row.get("neuterYn"),
        "processState": row.get("processState"),
        "specialMark": row.get("specialMark"),
        "happenPlace": row.get("happenPlace"),
        "noticeSdt": row.get("noticeSdt"),
        "noticeEdt": row.get("noticeEdt"),
        "popfile1": row.get("popfile1"),
        "careNm": row.get("careNm"),
        "careTel": row.get("careTel"),
        "careAddr": row.get("careAddr"),
        "orgNm": row.get("orgNm"),
    }


@tool
def search_pet_friendly_travel(
    region: str,
    category: str = "",
    page: int = 1,
) -> dict[str, Any]:
    """반려동물 동반 가능한 관광지·숙소·음식점 목록을 지역/카테고리 조건으로 검색한다.

    목록 API 이므로 동반 조건(동반유형·동반가능동물·필요사항)은 포함되지 않는다.
    동반 조건이 필요하면 이 Tool 의 결과에 있는 content_id 로 get_pet_travel_detail 을 호출한다.

    Args:
        region: 지역명. '강릉', '서울 성수동' 처럼 자연어 그대로 넘긴다.
        category: '관광지' / '숙박' / '음식점' / '레포츠' / '문화시설' / '쇼핑' 중 하나. 비우면 전체.
        page: 결과 페이지. 더 보여달라고 하면 2, 3 으로 올린다.

    동반 조건은 시스템이 상위 몇 건에 대해 자동으로 붙여 주므로 따로 요청하지 않아도 된다.
    """
    resolved_region = _resolved_region(region)
    if err := _need_region(resolved_region, "l_dong_regn_cd"):
        return err
    svc = get_services()
    try:
        rows = svc.client.area_based_list(
            l_dong_regn_cd=resolved_region["l_dong_regn_cd"],
            l_dong_signgu_cd=resolved_region.get("l_dong_signgu_cd"),
            content_type_id=CONTENT_TYPE.get(category.strip()) if category else None,
            num_of_rows=20,
            page_no=max(1, page),
        )
    except PublicDataError as exc:
        log.warning("search_pet_friendly_travel 실패: %s", exc)
        return {"error": "api_failed", "message": "현재 동반여행지 조회가 어렵습니다.", "items": []}

    return {
        "items": rows,
        "region": resolved_region.get("label") or region,
        "category": category,
        "total": len(rows),
        "page": page,
    }


@tool
def get_pet_travel_detail(content_ids: list[str]) -> dict[str, Any]:
    """장소의 반려동물 동반 조건(동반유형·동반가능동물·필요사항·사고대비사항)을 조회한다.

    search_pet_friendly_travel 결과의 상위 3~5건에 대해서만 호출한다.
    실측상 acmpyTypeCd 외의 필드는 비어 있는 경우가 많으며, 빈 값은 '미확인'으로 다룬다.

    Args:
        content_ids: 조회할 콘텐츠ID 목록. 최대 5개까지만 처리된다.
    """
    results = fetch_pet_details(content_ids)
    return {"items": results, "total": len(results)}


@tool
def save_user_preference(
    runtime: ToolRuntime,
    region: str = "",
    upkind: str = "",
    size: str = "",
    has_pet: bool | None = None,
) -> dict[str, Any]:
    """사용자의 선호 지역·품종·반려동물 보유 여부를 세션 간 장기 저장한다.

    사용자가 선호를 분명히 밝혔을 때만 호출하고, 값이 없는 항목은 비워 둔다.

    Args:
        region: 선호 지역. 예: '서울', '강원 강릉'
        upkind: 선호 축종. '개' / '고양이' / '기타'
        size: 선호 크기. '소형견' / '중형견' / '대형견'
        has_pet: 이미 반려동물을 기르고 있는지 여부

    user_id 는 Runtime Context 에서 읽으며 모델이 지정할 수 없다.
    """
    # dict 인자는 properties 없는 빈 오브젝트로 직렬화돼 strict 스키마에서 거부되므로
    # 원시 타입 인자로 받아 여기서 조립한다.
    incoming = {"region": region, "upkind": upkind, "size": size, "has_pet": has_pet}
    preferences = {k: v for k, v in incoming.items() if v not in ("", None)}
    if not preferences:
        return {"saved": False, "message": "저장할 선호 정보가 없습니다."}

    user_id = getattr(runtime.context, "user_id", None)
    if not user_id or runtime.store is None:
        return {"saved": False, "message": "저장소가 없어 이번 대화에서만 기억합니다."}
    try:
        current = runtime.store.get(("preferences",), user_id)
        merged = {**((current.value if current else {}) or {}), **preferences}
        runtime.store.put(("preferences",), user_id, merged)
        return {"saved": True, "preferences": merged}
    except Exception as exc:  # 저장 실패가 대화를 막지 않도록 한다
        log.warning("save_user_preference 실패: %s", exc)
        return {"saved": False, "message": "선호 정보를 저장하지 못했지만 대화는 계속됩니다."}


ADOPTION_TOOLS = [search_rescued_animals, get_animal_detail]
TRAVEL_TOOLS = [search_pet_friendly_travel, get_pet_travel_detail]
ALL_TOOLS = [*ADOPTION_TOOLS, *TRAVEL_TOOLS, save_user_preference]
