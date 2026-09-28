"""Replay helpers: spans from other recorders (JSON-string attributes, section records without roles)."""

from hone_models.replay import decoded, role_at

MESSAGES = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "A storm.\n\nFish."}]


def test_decoded_reads_json_strings_and_leaves_other_values() -> None:
    assert decoded('[{"id": "a"}]') == [{"id": "a"}]
    assert decoded('{"seed": 1}') == {"seed": 1}
    assert decoded("[not json") == "[not json"
    assert decoded("plain text") == "plain text"
    assert decoded([1]) == [1]


def test_role_at_finds_the_message_holding_a_character() -> None:
    assert role_at(MESSAGES, 0) == "system"
    assert role_at(MESSAGES, len("Be brief.\n\n")) == "user"
    assert role_at(MESSAGES, 10_000) == "user"
