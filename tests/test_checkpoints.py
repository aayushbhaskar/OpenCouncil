from __future__ import annotations

from open_council.runtime.checkpoints import ensure_thread_id, list_threads, touch_thread


def test_touch_and_list_threads(tmp_path) -> None:
    db = tmp_path / "checkpoints.sqlite"
    touch_thread(thread_id="t1", mode="odin", path=db)
    touch_thread(thread_id="t2", mode="artemis", path=db)
    rows = list_threads(path=db, limit=10)
    assert [row.thread_id for row in rows] == ["t2", "t1"]
    assert rows[0].mode == "artemis"


def test_ensure_thread_id_generates_when_blank() -> None:
    generated = ensure_thread_id("")
    assert len(generated) > 8
    assert ensure_thread_id("abc") == "abc"
