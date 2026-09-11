"""S-08 설계서 테스트 시나리오 — 상태 fan-out(병렬 상세조회)과 그 한계.

README '구현하면서 확인한 API 사실 1' — 장소 수만큼 detailPetTour2 호출이 늘어나므로
상위 3~5건만 '병렬' 호출한다는 설계 근거가 실제로 동시 실행되는지(순차 실행이 아닌지),
그리고 상한(detail_fanout)이 실제로 지켜지는지를 LLM/네트워크 없이 검증한다.
"""

import time
from types import SimpleNamespace

from petpal import tools
from petpal.api import PublicDataError


class SlowClient:
    """호출마다 지연이 있는 가짜 클라이언트. 순차 호출이면 총 시간이 N * delay 에 가깝다."""

    def __init__(self, delay: float):
        self.delay = delay
        self.calls: list[str] = []

    def detail_pet_tour(self, content_id):
        self.calls.append(content_id)
        time.sleep(self.delay)
        return {"acmpyTypeCd": "전구역 동반가능"}


def test_detail_fetch_runs_concurrently_not_sequentially(monkeypatch):
    client = SlowClient(delay=0.08)
    monkeypatch.setattr(tools, "get_services",
                        lambda: SimpleNamespace(client=client, settings=SimpleNamespace(detail_fanout=5)))

    ids = [f"C{i}" for i in range(5)]
    started = time.monotonic()
    result = tools.fetch_pet_details(ids)
    elapsed = time.monotonic() - started

    assert set(result) == set(ids)
    serial_time = client.delay * len(ids)
    assert elapsed < serial_time * 0.6, (
        f"{elapsed:.3f}s 소요 — 순차 실행({serial_time:.3f}s)에 가까워 병렬화가 안 됐을 수 있다"
    )


def test_detail_fetch_is_capped_at_configured_fanout(monkeypatch):
    """상위 detail_fanout 건까지만 상세조회한다(설계서 2.2 6단계) — 장소가 많아도 무한정 늘지 않는다."""
    client = SlowClient(delay=0.0)
    monkeypatch.setattr(tools, "get_services",
                        lambda: SimpleNamespace(client=client, settings=SimpleNamespace(detail_fanout=3)))

    tools.fetch_pet_details([f"C{i}" for i in range(10)])
    assert len(client.calls) == 3


def test_one_slow_or_failed_detail_call_does_not_break_the_others(monkeypatch):
    """fan-out 중 일부가 실패해도 나머지 결과는 정상적으로 모여야 한다."""

    class FlakyClient:
        def detail_pet_tour(self, content_id):
            if content_id == "C1":
                raise PublicDataError("timeout")
            return {"acmpyTypeCd": "전구역 동반가능"}

    monkeypatch.setattr(tools, "get_services",
                        lambda: SimpleNamespace(client=FlakyClient(), settings=SimpleNamespace(detail_fanout=5)))

    result = tools.fetch_pet_details(["C1", "C2"])
    assert result["C1"] == {}
    assert result["C2"]["acmpyTypeCd"] == "전구역 동반가능"
