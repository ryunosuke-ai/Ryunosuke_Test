"""4主ラベル化後の分類・推定・遷移ロジックのテスト"""

import logging
from types import SimpleNamespace

from bayes_engine import CLASSIFY_ACTION_PROMPT, classify_action, judge_memory_signal, update_posterior
from models import ActionType, Observation, Phase
from phase_manager import PhaseManager


def _make_client(response_text: str):
    message = SimpleNamespace(content=response_text)
    choice = SimpleNamespace(message=message)
    completions = SimpleNamespace(create=lambda **kwargs: SimpleNamespace(choices=[choice]))
    chat = SimpleNamespace(completions=completions)
    return SimpleNamespace(chat=chat)


def _make_failing_client():
    def _raise(**kwargs):
        raise RuntimeError("LLM unavailable")

    completions = SimpleNamespace(create=_raise)
    chat = SimpleNamespace(completions=completions)
    return SimpleNamespace(chat=chat)


def test_classify_action_parses_json_response():
    client = _make_client('{"primary_label":"ACTIVE","reason":"自発的に話を広げているため。"}')

    result = classify_action(client, "dummy", "祖父との思い出を詳しく話します", logging.getLogger("test"))

    assert result.action_type == ActionType.ACTIVE
    assert result.reason == "自発的に話を広げているため。"


def test_classify_action_falls_back_to_label_text():
    client = _make_client("MINIMAL")

    result = classify_action(client, "dummy", "うーん", logging.getLogger("test"))

    assert result.action_type == ActionType.MINIMAL


def test_classify_action_returns_responsive_on_failure():
    client = _make_failing_client()

    result = classify_action(client, "dummy", "はい", logging.getLogger("test"))

    assert result.action_type == ActionType.RESPONSIVE


def test_judge_memory_signal_parses_json_response():
    client = _make_client('{"memory_flag": true, "note": "過去の具体的エピソードです"}')

    memory_flag, note = judge_memory_signal(client, "dummy", "子どもの頃、祖父と公園に行きました", logging.getLogger("test"))

    assert memory_flag is True
    assert note == "過去の具体的エピソードです"


def test_update_posterior_follows_label_order():
    prior = 0.5

    posterior_disengage = update_posterior(prior, ActionType.DISENGAGE)
    posterior_minimal = update_posterior(prior, ActionType.MINIMAL)
    posterior_responsive = update_posterior(prior, ActionType.RESPONSIVE)
    posterior_active = update_posterior(prior, ActionType.ACTIVE)

    assert posterior_disengage < posterior_minimal < posterior_responsive < posterior_active


def test_classify_prompt_contains_boundary_rules_and_output_constraints():
    assert "MINIMAL と RESPONSIVE の違い" in CLASSIFY_ACTION_PROMPT
    assert "RESPONSIVE と ACTIVE の違い" in CLASSIFY_ACTION_PROMPT
    assert "MINIMAL と DISENGAGE の違い" in CLASSIFY_ACTION_PROMPT
    assert "\"primary_label\"" in CLASSIFY_ACTION_PROMPT
    assert "\"reason\"" in CLASSIFY_ACTION_PROMPT
    assert "ACTIVE" in CLASSIFY_ACTION_PROMPT
    assert "RESPONSIVE" in CLASSIFY_ACTION_PROMPT
    assert "MINIMAL" in CLASSIFY_ACTION_PROMPT
    assert "DISENGAGE" in CLASSIFY_ACTION_PROMPT


def test_bridge_moves_to_deep_dive_on_active():
    mgr = PhaseManager(logger=logging.getLogger("test"))
    mgr.phase = Phase.BRIDGE

    obs = Observation(user_text="それで思い出したんですが、昔よく行っていたんです", action_type=ActionType.ACTIVE)
    mgr.transition_policy(obs, p_want_talk=0.6)

    assert mgr.phase == Phase.DEEP_DIVE


def test_bridge_moves_to_ending_on_disengage_with_low_probability():
    mgr = PhaseManager(logger=logging.getLogger("test"))
    mgr.phase = Phase.BRIDGE

    obs = Observation(user_text="その話はしたくないです", action_type=ActionType.DISENGAGE)
    mgr.transition_policy(obs, p_want_talk=0.2)

    assert mgr.phase == Phase.ENDING


def test_deep_dive_stays_on_responsive():
    mgr = PhaseManager(logger=logging.getLogger("test"))
    mgr.phase = Phase.DEEP_DIVE

    obs = Observation(user_text="学生の頃です", action_type=ActionType.RESPONSIVE)
    mgr.transition_policy(obs, p_want_talk=0.6)

    assert mgr.phase == Phase.DEEP_DIVE
    assert mgr.deep_drop_count == 0


def test_deep_dive_returns_to_surroundings_after_two_minimal_replies():
    mgr = PhaseManager(logger=logging.getLogger("test"))
    mgr.phase = Phase.DEEP_DIVE

    obs = Observation(user_text="うーん", action_type=ActionType.MINIMAL)
    mgr.transition_policy(obs, p_want_talk=0.5)
    assert mgr.phase == Phase.DEEP_DIVE

    mgr.transition_policy(obs, p_want_talk=0.5)
    assert mgr.phase == Phase.SURROUNDINGS
