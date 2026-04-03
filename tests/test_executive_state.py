import pytest

from open_council.state.executive import initialize_odin_state


def test_initialize_odin_state_builds_expected_shape() -> None:
    state = initialize_odin_state("design a resilient cache")

    assert state["query"] == "design a resilient cache"
    assert state["parallel_drafts"] == []
    assert "final_synthesis" not in state


def test_initialize_odin_state_rejects_empty_query() -> None:
    with pytest.raises(ValueError):
        initialize_odin_state("   ")
