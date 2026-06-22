from piper_smolvla.rollout_preview import is_quit_key, is_start_or_pause_key


def test_rollout_preview_start_pause_keys():
    assert is_start_or_pause_key(ord(" "))
    assert is_start_or_pause_key(10)
    assert is_start_or_pause_key(13)
    assert not is_start_or_pause_key(ord("q"))


def test_rollout_preview_quit_keys():
    assert is_quit_key(ord("q"))
    assert is_quit_key(ord("Q"))
    assert is_quit_key(27)
    assert not is_quit_key(ord(" "))
