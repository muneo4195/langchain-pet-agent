"""설계서 1.2 기능요건 6 / 3.1 Store — 세션을 넘어 값이 남는지."""

from petpal.persistence import build_checkpointer, build_persistence, build_store


def test_store_survives_reopen(tmp_path):
    """연결을 닫고 다시 열어도 값이 남는다(= 프로세스를 껐다 켜도 남는다)."""
    db = tmp_path / "store.sqlite"

    first = build_store(db)
    first.put(("preferences",), "u1", {"region": "강릉", "size": "소형견"})

    second = build_store(db)  # 새 연결 = 새 프로세스와 같은 상황
    item = second.get(("preferences",), "u1")
    assert item is not None and item.value == {"region": "강릉", "size": "소형견"}


def test_store_is_per_user(tmp_path):
    db = tmp_path / "store.sqlite"
    store = build_store(db)
    store.put(("preferences",), "u1", {"region": "서울"})
    assert store.get(("preferences",), "u2") is None


def test_checkpointer_file_is_created(tmp_path):
    db = tmp_path / "cp.sqlite"
    build_checkpointer(db)
    assert db.exists()


def test_in_memory_mode_is_opt_in():
    """테스트용으로 메모리 저장을 고를 수 있다."""
    checkpointer, store = build_persistence(in_memory=True)
    assert type(checkpointer).__name__ == "InMemorySaver"
    assert type(store).__name__ == "InMemoryStore"
