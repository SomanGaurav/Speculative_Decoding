from specdec.adaptive_length import LengthConfig, should_continue_drafting


def test_fixed_k_always_continues_until_cap():
    config = LengthConfig(mode="fixed_k", max_k=5)
    for step in range(4):
        assert should_continue_drafting(step, entropy=999.0, config=config)
    assert not should_continue_drafting(4, entropy=0.0, config=config)  # hit max_k


def test_entropy_threshold_stops_on_high_entropy():
    config = LengthConfig(mode="entropy_threshold", max_k=10, entropy_threshold=1.0)
    assert should_continue_drafting(0, entropy=0.5, config=config)
    assert not should_continue_drafting(1, entropy=1.5, config=config)


def test_entropy_threshold_respects_max_k_even_if_confident():
    config = LengthConfig(mode="entropy_threshold", max_k=2, entropy_threshold=100.0)
    assert should_continue_drafting(0, entropy=0.0, config=config)
    assert not should_continue_drafting(1, entropy=0.0, config=config)


def test_unknown_mode_raises():
    import pytest

    config = LengthConfig(mode="bogus")
    with pytest.raises(ValueError):
        should_continue_drafting(0, entropy=0.0, config=config)
