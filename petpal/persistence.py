"""영속 저장 — 설계서 1.2 기능요건 6 / 3.1 Store.

InMemoryStore·InMemorySaver 는 프로세스와 함께 사라져서 "세션을 넘어 영속"이라는
설계를 만족하지 못한다. SQLite 파일에 담아 CLI 를 껐다 켜도 남도록 한다.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

log = logging.getLogger("petpal.persistence")

DATA_DIR = Path(__file__).resolve().parent.parent / ".data"
CHECKPOINT_DB = DATA_DIR / "checkpoints.sqlite"
STORE_DB = DATA_DIR / "store.sqlite"


def _connect(path: Path, *, autocommit: bool = False) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    # LangGraph 는 워커 스레드에서도 접근하므로 check_same_thread 를 끈다.
    # SqliteStore 는 BEGIN 을 직접 실행해서 파이썬 기본 트랜잭션 관리와 충돌한다
    # ("cannot start a transaction within a transaction") → autocommit 으로 연다.
    return sqlite3.connect(
        str(path),
        check_same_thread=False,
        isolation_level=None if autocommit else "",
    )


def _serializer():
    """AgentResponse 를 체크포인트에 담을 수 있도록 직렬화 허용 목록에 등록한다.

    등록하지 않으면 복원 시 "Deserializing unregistered type" 경고가 뜨고,
    향후 버전에서는 아예 차단된다.
    """
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    from .schemas import AgentResponse, AnimalCard, PetTravelCard

    return JsonPlusSerializer().with_msgpack_allowlist(
        [AgentResponse, AnimalCard, PetTravelCard]
    )


def build_checkpointer(path: Path | None = None):
    """대화 이력(State)을 파일에 보존한다."""
    from langgraph.checkpoint.sqlite import SqliteSaver

    saver = SqliteSaver(_connect(path or CHECKPOINT_DB), serde=_serializer())
    saver.setup()
    return saver


def build_store(path: Path | None = None):
    """user_preference 같은 세션 초월 값을 파일에 보존한다."""
    from langgraph.store.sqlite import SqliteStore

    store = SqliteStore(_connect(path or STORE_DB, autocommit=True))
    store.setup()
    return store


def build_persistence(in_memory: bool = False):
    """(checkpointer, store) 한 쌍. 실패하면 메모리 저장으로 떨어뜨린다."""
    if in_memory:
        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.store.memory import InMemoryStore

        return InMemorySaver(), InMemoryStore()
    try:
        return build_checkpointer(), build_store()
    except Exception as exc:  # 디스크 문제로 대화 자체가 막히지 않도록
        log.warning("영속 저장 초기화 실패, 메모리 저장으로 진행합니다: %s", exc)
        return build_persistence(in_memory=True)
