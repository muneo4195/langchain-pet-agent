"""공공데이터 OpenAPI 클라이언트.

설계서 1.5 안정성: 타임아웃 시 지수 백오프로 3회 재시도한 뒤 실패를 알린다.
설계서 1.5 보안: 응답 봉투(envelope)를 검증한 뒤에만 본문을 사용한다.
"""

from __future__ import annotations

import logging
import random
import time
from datetime import date, timedelta
from typing import Any
from urllib.parse import quote, urlencode

import requests

from .config import ANIMAL_BASE, TRAVEL_BASE, Settings

log = logging.getLogger("petpal.api")

# 정상 응답 코드 — 구조동물은 "00", 관광공사는 "0000" 을 쓴다.
OK_CODES = {"00", "0000"}


class PublicDataError(RuntimeError):
    """공공 API 호출 실패. Tool 계층에서 사용자 안내 메시지로 변환한다."""


def _encoded_key(raw: str) -> str:
    """.env 에 인코딩 키/디코딩 키 어느 쪽을 넣어도 동작하게 맞춘다."""
    return raw if "%" in raw else quote(raw, safe="")


def _rows(body: dict[str, Any]) -> list[dict[str, Any]]:
    """items 가 ''(0건) · dict(1건) · list 로 제각각 오는 것을 리스트로 정규화한다."""
    items = body.get("items")
    if not items or isinstance(items, str):
        return []
    item = items.get("item") if isinstance(items, dict) else items
    if not item:
        return []
    return item if isinstance(item, list) else [item]


class PublicDataClient:
    def __init__(self, settings: Settings, session: requests.Session | None = None) -> None:
        self.s = settings
        self.session = session or requests.Session()
        self._key = _encoded_key(settings.service_key)

    # ------------------------------------------------------------------ 공통
    def _get(self, base: str, path: str, **params: Any) -> dict[str, Any]:
        query = urlencode({k: v for k, v in params.items() if v not in (None, "")})
        url = f"{base}/{path}?serviceKey={self._key}&_type=json&{query}"
        last: Exception | None = None

        for attempt in range(1, self.s.max_retries + 1):
            try:
                res = self.session.get(url, timeout=self.s.http_timeout)
                res.raise_for_status()
                payload = res.json()
                return self._unwrap(payload, path)
            except (requests.Timeout, requests.ConnectionError, ValueError) as exc:
                last = exc
                if attempt < self.s.max_retries:
                    delay = self.s.backoff_factor ** (attempt - 1) * (1 + random.random() * 0.3)
                    log.warning("%s 재시도 %d/%d (%s)", path, attempt, self.s.max_retries, exc)
                    time.sleep(delay)
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else 0
                last = exc
                retryable = status in {408, 429} or status >= 500
                if not retryable or attempt == self.s.max_retries:
                    break
                time.sleep(self.s.backoff_factor ** (attempt - 1))

        raise PublicDataError(f"{path} 호출에 실패했습니다: {last}") from last

    @staticmethod
    def _unwrap(payload: dict[str, Any], path: str) -> dict[str, Any]:
        if "OpenAPI_ServiceResponse" in payload:  # 인증키·경로 오류는 이 형태로 온다
            head = payload["OpenAPI_ServiceResponse"].get("cmmMsgHeader", {})
            raise PublicDataError(f"{path}: {head.get('errMsg')} / {head.get('returnAuthMsg')}")
        response = payload.get("response") or {}
        header, body = response.get("header") or {}, response.get("body") or {}
        code = str(header.get("resultCode", ""))
        if code not in OK_CODES:
            raise PublicDataError(f"{path}: resultCode={code} {header.get('resultMsg')}")
        return body

    # -------------------------------------------------------------- 구조동물
    def abandonment_public(
        self,
        *,
        upr_cd: str | None = None,
        org_cd: str | None = None,
        upkind: str | None = None,
        kind: str | None = None,
        state: str | None = None,
        neuter_yn: str | None = None,
        bgnde: str | None = None,
        endde: str | None = None,
        num_of_rows: int = 50,
        page_no: int = 1,
    ) -> list[dict[str, Any]]:
        """구조동물 공고 검색. bgnde/endde 는 생략 시 최근 N일로 채운다."""
        if not endde:
            endde = date.today().strftime("%Y%m%d")
        if not bgnde:
            bgnde = (date.today() - timedelta(days=self.s.default_search_days)).strftime("%Y%m%d")
        body = self._get(
            ANIMAL_BASE, "abandonmentPublic_v2",
            upr_cd=upr_cd, org_cd=org_cd, upkind=upkind, kind=kind,
            state=state, neuter_yn=neuter_yn, bgnde=bgnde, endde=endde,
            numOfRows=num_of_rows, pageNo=page_no,
        )
        return _rows(body)

    def animal_sido(self) -> list[dict[str, Any]]:
        return _rows(self._get(ANIMAL_BASE, "sido_v2", numOfRows=100, pageNo=1))

    def animal_sigungu(self, upr_cd: str) -> list[dict[str, Any]]:
        return _rows(self._get(ANIMAL_BASE, "sigungu_v2", upr_cd=upr_cd, numOfRows=200, pageNo=1))

    def animal_kinds(self, up_kind_cd: str) -> list[dict[str, Any]]:
        return _rows(self._get(ANIMAL_BASE, "kind_v2", up_kind_cd=up_kind_cd, numOfRows=500, pageNo=1))

    # -------------------------------------------------------------- 동반여행
    def area_based_list(
        self,
        *,
        l_dong_regn_cd: str,
        l_dong_signgu_cd: str | None = None,
        content_type_id: str | None = None,
        arrange: str = "A",
        num_of_rows: int = 20,
        page_no: int = 1,
    ) -> list[dict[str, Any]]:
        """동반여행지 목록. 동반 조건은 포함되지 않으므로 detail_pet_tour 로 보강해야 한다."""
        body = self._get(
            TRAVEL_BASE, "areaBasedList2",
            MobileOS="ETC", MobileApp="petpal",
            lDongRegnCd=l_dong_regn_cd, lDongSignguCd=l_dong_signgu_cd,
            contentTypeId=content_type_id, arrange=arrange,
            numOfRows=num_of_rows, pageNo=page_no,
        )
        return _rows(body)

    def detail_pet_tour(self, content_id: str) -> dict[str, Any]:
        """장소 1건의 반려동물 동반 조건. 대부분의 필드가 빈 문자열로 오는 점에 유의."""
        body = self._get(
            TRAVEL_BASE, "detailPetTour2",
            MobileOS="ETC", MobileApp="petpal",
            contentId=content_id, numOfRows=10, pageNo=1,
        )
        rows = _rows(body)
        return rows[0] if rows else {}

    def travel_ldong_code(self, l_dong_regn_cd: str | None = None) -> list[dict[str, Any]]:
        """시도(인자 없음) 또는 해당 시도의 시군구 법정동 코드 목록."""
        body = self._get(
            TRAVEL_BASE, "ldongCode2",
            MobileOS="ETC", MobileApp="petpal",
            lDongRegnCd=l_dong_regn_cd, numOfRows=300, pageNo=1,
        )
        return _rows(body)
