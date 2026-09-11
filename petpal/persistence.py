"""영속 저장 — 설계서 1.2 기능요건 6 / 3.1 Store.

InMemoryStore·InMemorySaver 는 프로세스와 함께 사라져서 "세션을 넘어 영속"이라는
설계를 만족하지 못한다. SQLite 파일에 담아 CLI 를 껐다 켜도 남도록 한다.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
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

    # with_msgpack_allowlist 는 기본값이 "전체 허용(True)"일 때 병합을 건너뛰므로
    # 생성자에 직접 넘겨 명시 목록으로 고정한다.
    return JsonPlusSerializer(
        allowed_msgpack_modules=[AgentResponse, AnimalCard, PetTravelCard]
    )


def build_checkpointer(path: Path | None = None):
    """대화 이력(State)을 파일에 보존한다."""
    from langgraph.checkpoint.sqlite import SqliteSaver

    saver = SqliteSaver(_connect(path or CHECKPOINT_DB), serde=_serializer())
    saver.setup()
    return saver


def build_store(path: Path | None = None):
    """user_preference 같은 세션 초월 값을 파일에 보존한다."""
    try:
        from langgraph.store.sqlite import SqliteStore  # type: ignore
    except Exception:
        SqliteStore = None  # type: ignore[assignment]

    conn = _connect(path or STORE_DB, autocommit=True)
    if SqliteStore is not None:
        store = SqliteStore(conn)
        store.setup()
        return store

    # LangGraph 배포 조합에 따라 sqlite store 어댑터가 별도 패키지로 빠져 있을 수 있다.
    # 이 프로젝트는 기본 기능(put/get)만 있으면 되므로, 최소 SQLite 어댑터를 제공한다.
    from langgraph.store.base import BaseStore, GetOp, Item, ListNamespacesOp, PutOp, SearchItem, SearchOp

    class SqliteStoreCompat(BaseStore):
        def __init__(self, connection: sqlite3.Connection):
            self._conn = connection
            self.setup()

        def setup(self) -> None:
            cur = self._conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS kv_store (
                    namespace TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (namespace, key)
                )
                """
            )
            self._conn.commit()

        @staticmethod
        def _ns_to_str(ns: tuple[str, ...]) -> str:
            return "/".join(ns)

        @staticmethod
        def _now() -> str:
            return datetime.now(timezone.utc).isoformat()

        def batch(self, ops):
            results = []
            cur = self._conn.cursor()
            for op in ops:
                if isinstance(op, GetOp):
                    ns = self._ns_to_str(op.namespace)
                    row = cur.execute(
                        "SELECT value_json, created_at, updated_at FROM kv_store WHERE namespace=? AND key=?",
                        (ns, op.key),
                    ).fetchone()
                    if not row:
                        results.append(None)
                        continue
                    value_json, created_at, updated_at = row
                    results.append(
                        Item(
                            namespace=op.namespace,
                            key=op.key,
                            value=json.loads(value_json),
                            created_at=created_at,
                            updated_at=updated_at,
                        )
                    )
                    continue

                if isinstance(op, PutOp):
                    ns = self._ns_to_str(op.namespace)
                    now = self._now()
                    if op.value is None:
                        cur.execute(
                            "DELETE FROM kv_store WHERE namespace=? AND key=?",
                            (ns, op.key),
                        )
                        results.append(None)
                        continue

                    existing = cur.execute(
                        "SELECT created_at FROM kv_store WHERE namespace=? AND key=?",
                        (ns, op.key),
                    ).fetchone()
                    created_at = existing[0] if existing else now
                    cur.execute(
                        """
                        INSERT INTO kv_store(namespace, key, value_json, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT(namespace, key) DO UPDATE SET
                            value_json=excluded.value_json,
                            updated_at=excluded.updated_at
                        """,
                        (ns, op.key, json.dumps(op.value, ensure_ascii=False), created_at, now),
                    )
                    results.append(None)
                    continue

                if isinstance(op, SearchOp):
                    # Simple prefix search + optional exact-match filters. No vector search.
                    prefix = self._ns_to_str(op.namespace_prefix)
                    like = prefix + "%" if prefix else "%"
                    rows = cur.execute(
                        "SELECT namespace, key, value_json, created_at, updated_at FROM kv_store WHERE namespace LIKE ?",
                        (like,),
                    ).fetchall()
                    items: list[SearchItem] = []
                    for ns_str, key, value_json, created_at, updated_at in rows:
                        value = json.loads(value_json)
                        if op.filter:
                            ok = True
                            for fk, fv in op.filter.items():
                                if value.get(fk) != fv:
                                    ok = False
                                    break
                            if not ok:
                                continue
                        ns_tuple = tuple(ns_str.split("/")) if ns_str else tuple()
                        items.append(
                            SearchItem(
                                namespace=ns_tuple,
                                key=key,
                                value=value,
                                created_at=created_at,
                                updated_at=updated_at,
                                score=None,
                            )
                        )
                    results.append(items[op.offset : op.offset + op.limit])
                    continue

                if isinstance(op, ListNamespacesOp):
                    rows = cur.execute(
                        "SELECT DISTINCT namespace FROM kv_store",
                    ).fetchall()
                    nss = []
                    for (ns_str,) in rows:
                        ns_tuple = tuple(ns_str.split("/")) if ns_str else tuple()
                        if op.max_depth is not None:
                            ns_tuple = ns_tuple[: op.max_depth]
                        nss.append(ns_tuple)
                    results.append(nss[op.offset : op.offset + op.limit])
                    continue

                raise TypeError(f"Unsupported store op: {type(op)}")
            self._conn.commit()
            return results

        async def abatch(self, ops):
            # This adapter does synchronous sqlite3 I/O; keep the async surface for compatibility.
            return self.batch(ops)

    return SqliteStoreCompat(conn)


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
