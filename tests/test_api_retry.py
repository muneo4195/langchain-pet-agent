"""설계서 1.5 안정성 / TS-02 — 타임아웃 재시도와 Fallback. 네트워크 없이 검증한다."""

import json

import pytest
import requests

from petpal.api import PublicDataClient, PublicDataError, _rows
from petpal.config import Settings

SETTINGS = Settings(service_key="test-key", max_retries=3, backoff_factor=0.0, http_timeout=0.1)


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)

    def json(self):
        return self._payload


class FakeSession:
    """지정한 순서대로 예외를 던지거나 응답을 돌려주는 세션."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def get(self, url, timeout=None):
        self.calls += 1
        outcome = self.outcomes.pop(0) if self.outcomes else self.outcomes
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


OK_BODY = {"response": {"header": {"resultCode": "00", "resultMsg": "NORMAL SERVICE."},
                        "body": {"items": {"item": [{"desertionNo": "A1"}]}}}}


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr("petpal.api.time.sleep", lambda _: None)


def test_succeeds_after_transient_timeouts():
    """타임아웃 2회 뒤 성공하면 결과를 정상 반환한다."""
    session = FakeSession(requests.Timeout("t1"), requests.Timeout("t2"), FakeResponse(OK_BODY))
    client = PublicDataClient(SETTINGS, session=session)
    rows = client.abandonment_public(upr_cd="6110000")
    assert session.calls == 3
    assert rows[0]["desertionNo"] == "A1"


def test_raises_after_max_retries():
    """3회 모두 실패하면 PublicDataError 로 올린다(Tool 이 안내 메시지로 변환)."""
    session = FakeSession(*(requests.Timeout("t") for _ in range(3)))
    client = PublicDataClient(SETTINGS, session=session)
    with pytest.raises(PublicDataError):
        client.abandonment_public(upr_cd="6110000")
    assert session.calls == 3, "max_retries 를 넘겨 호출하면 안 된다"


def test_tool_converts_failure_to_message(monkeypatch):
    """TS-02 Pass 기준 — 무한루프 없이 종료하고 사용자에게 안내 메시지를 준다."""
    from petpal import tools

    class Boom:
        def abandonment_public(self, **_):
            raise PublicDataError("timeout")

    monkeypatch.setattr(tools, "get_services",
                        lambda: type("S", (), {"client": Boom(), "codes": None, "settings": SETTINGS})())
    monkeypatch.setattr(tools, "_resolved_region", lambda _: {"upr_cd": "6110000", "label": "서울"})
    out = tools.search_rescued_animals.invoke(
        {"name": "search_rescued_animals", "args": {"region": "서울"}, "id": "1", "type": "tool_call"})
    body = json.loads(out.content)
    assert body["error"] == "api_failed" and body["items"] == []
    assert "조회가 어렵" in body["message"]


def test_client_error_is_not_retried():
    """4xx 는 재시도해도 소용없으므로 즉시 실패시킨다."""
    session = FakeSession(FakeResponse({}, status=400))
    client = PublicDataClient(SETTINGS, session=session)
    with pytest.raises(PublicDataError):
        client.animal_sido()
    assert session.calls == 1


def test_auth_error_envelope_is_surfaced():
    """인증키 오류는 별도 봉투로 와서 resultCode 검사에 걸리지 않는다."""
    payload = {"OpenAPI_ServiceResponse": {"cmmMsgHeader": {
        "errMsg": "SERVICE_KEY_IS_NOT_REGISTERED_ERROR", "returnAuthMsg": "등록되지 않은 서비스키"}}}
    client = PublicDataClient(SETTINGS, session=FakeSession(FakeResponse(payload)))
    with pytest.raises(PublicDataError, match="등록되지 않은 서비스키"):
        client.animal_sido()


@pytest.mark.parametrize("items,expected", [
    ({"items": ""}, 0),                                  # 0건이면 빈 문자열로 온다
    ({"items": {"item": {"desertionNo": "A1"}}}, 1),     # 1건이면 dict 로 올 수 있다
    ({"items": {"item": [{"a": 1}, {"a": 2}]}}, 2),
    ({}, 0),
])
def test_rows_normalizes_shapes(items, expected):
    """지자체별로 items 모양이 제각각이라 방어적으로 정규화한다."""
    assert len(_rows(items)) == expected
