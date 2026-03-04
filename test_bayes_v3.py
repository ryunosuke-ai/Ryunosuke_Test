"""
bayes_v3.py のユニットテスト
azure-cognitiveservices-speech / cv2 はモック化（未インストール）
OpenAI API 呼び出しもモック化してコスト不要でテスト
"""
import os
import sys
import tempfile
import logging
import unittest
from unittest.mock import MagicMock, patch

# ----------------------------------------------------------------
# 未インストールの外部ライブラリをモック化
# ----------------------------------------------------------------
_orig_stdout = sys.stdout  # bayes_v3 が stdout を書き換えるため保存

for _mod in [
    "azure",
    "azure.cognitiveservices",
    "azure.cognitiveservices.speech",
    "azure.cognitiveservices.speech.audio",
    "cv2",
]:
    sys.modules.setdefault(_mod, MagicMock())

# io.TextIOWrapper をパッチして stdout の置き換えを無効化
with patch("io.TextIOWrapper", side_effect=lambda *a, **kw: _orig_stdout), \
     patch("azure.cognitiveservices.speech.SpeechConfig"), \
     patch("azure.cognitiveservices.speech.SpeechRecognizer"), \
     patch("azure.cognitiveservices.speech.SpeechSynthesizer"), \
     patch("openai.AzureOpenAI"), \
     patch("cv2.imread", return_value=None):
    import bayes_v3
    from bayes_v3 import (
        MultimodalAgent, ActionType, Phase,
        Observation, MemoryUpdate, PhaseConfig,
    )
    from phase_manager import PhaseManager

sys.stdout = _orig_stdout  # 念のため復元


# ----------------------------------------------------------------
# テスト用エージェントファクトリ（API 呼び出しなし）
# ----------------------------------------------------------------
def make_agent(tmp_dir: str) -> MultimodalAgent:
    agent = MultimodalAgent.__new__(MultimodalAgent)

    agent.static_image_path = "experiment_image.jpg"
    agent.static_image_b64 = None
    agent.speech_key = "dummy"
    agent.speech_region = "dummy"
    agent.openai_key = "dummy"
    agent.openai_endpoint = "https://dummy/"
    agent.deployment_name = "dummy"
    agent.openai_api_version = "2024-02-15-preview"
    agent.run_dir = tmp_dir
    agent.history_file = os.path.join(tmp_dir, "log_test.txt")
    agent.analysis_csv = os.path.join(tmp_dir, "analysis_test.csv")
    agent.openai_client = MagicMock()
    agent.logger = logging.getLogger("test_bayes_agent")

    # PhaseManager を先に作成（プロパティ委譲のため必須）
    test_configs = {
        Phase.SETUP:        PhaseConfig(name=Phase.SETUP,        instruction="setup",  require_image=False, max_turns=1),
        Phase.INTRO:        PhaseConfig(name=Phase.INTRO,        instruction="intro",  require_image=False, max_turns=2),
        Phase.SURROUNDINGS: PhaseConfig(name=Phase.SURROUNDINGS, instruction="surr",   require_image=True,  max_turns=3),
        Phase.BRIDGE:       PhaseConfig(name=Phase.BRIDGE,       instruction="bridge", require_image=True,  max_turns=4),
        Phase.DEEP_DIVE:    PhaseConfig(name=Phase.DEEP_DIVE,    instruction="deep",   require_image=False, max_turns=6),
        Phase.ENDING:       PhaseConfig(name=Phase.ENDING,       instruction="ending", require_image=False, max_turns=2),
    }
    agent.phase_mgr = PhaseManager(phase_configs=test_configs, logger=agent.logger)

    agent.total_turns = 0
    agent.p_want_talk = 0.5

    agent.likelihoods = {
        "H1": {ActionType.SILENCE: 0.10, ActionType.NORMAL: 0.25, ActionType.DISCLOSURE: 0.65},
        "H0": {ActionType.SILENCE: 0.45, ActionType.NORMAL: 0.50, ActionType.DISCLOSURE: 0.05},
    }
    agent.minimal_normal_likelihood = {"H1": 0.05, "H0": 0.70}

    agent.asked_initial_image_question = False
    agent.conv_memory = MemoryUpdate(summary="", do_not_ask=[])
    agent.force_end = False
    agent.max_total_turns = 15
    return agent


def obs(action_type, memory_flag=False, minimal_reply=False, user_text="テスト"):
    return Observation(
        user_text=user_text,
        action_type=action_type,
        memory_flag=memory_flag,
        minimal_reply=minimal_reply,
    )


# ================================================================
# ベイズ更新
# ================================================================
class TestUpdatePosterior(unittest.TestCase):

    def setUp(self):
        self.agent = make_agent(tempfile.mkdtemp())

    def test_disclosure_raises_probability(self):
        self.agent.p_want_talk = 0.5
        self.assertGreater(self.agent.update_posterior(ActionType.DISCLOSURE), 0.5)

    def test_silence_lowers_probability(self):
        self.agent.p_want_talk = 0.5
        self.assertLess(self.agent.update_posterior(ActionType.SILENCE), 0.5)

    def test_minimal_reply_lowers_more_than_normal(self):
        self.agent.p_want_talk = 0.5
        p_normal = self.agent.update_posterior(ActionType.NORMAL, minimal_reply=False)
        self.agent.p_want_talk = 0.5
        p_minimal = self.agent.update_posterior(ActionType.NORMAL, minimal_reply=True)
        self.assertLess(p_minimal, p_normal)

    def test_probability_bounded(self):
        self.agent.p_want_talk = 0.001
        p = self.agent.update_posterior(ActionType.DISCLOSURE)
        self.assertGreaterEqual(p, 0.001)
        self.assertLessEqual(p, 0.999)

    def test_repeated_disclosure_approaches_1(self):
        self.agent.p_want_talk = 0.5
        for _ in range(20):
            self.agent.update_posterior(ActionType.DISCLOSURE)
        self.assertGreater(self.agent.p_want_talk, 0.90)

    def test_repeated_silence_approaches_0(self):
        self.agent.p_want_talk = 0.5
        for _ in range(20):
            self.agent.update_posterior(ActionType.SILENCE)
        self.assertLess(self.agent.p_want_talk, 0.10)

    def test_normalization_matches_bayes_formula(self):
        """P(H1|obs) がベイズ公式と一致するか。"""
        prior = 0.5
        self.agent.p_want_talk = prior
        action = ActionType.DISCLOSURE
        l_h1 = self.agent.likelihoods["H1"][action]
        l_h0 = self.agent.likelihoods["H0"][action]
        expected = (l_h1 * prior) / (l_h1 * prior + l_h0 * (1 - prior))
        self.assertAlmostEqual(self.agent.update_posterior(action), expected, places=5)

    def test_all_actions_stay_in_bounds(self):
        for prior in [0.1, 0.3, 0.5, 0.7, 0.9]:
            for action in ActionType:
                self.agent.p_want_talk = prior
                p = self.agent.update_posterior(action)
                self.assertGreaterEqual(p, 0.001)
                self.assertLessEqual(p, 0.999)


# ================================================================
# 生返事検出
# ================================================================
class TestIsMinimalReply(unittest.TestCase):

    def setUp(self):
        self.agent = make_agent(tempfile.mkdtemp())

    def _chk(self, text, expected):
        self.assertEqual(self.agent._is_minimal_reply(text), expected, f"'{text}'")

    def test_none(self):         self._chk(None, False)
    def test_empty(self):        self._chk("", False)
    def test_hai(self):          self._chk("はい", True)
    def test_soudesune(self):    self._chk("そうですね", True)
    def test_arigatou(self):     self._chk("ありがとうございます", True)
    def test_naruhodone(self):   self._chk("なるほど", True)
    def test_un(self):           self._chk("うん", True)
    def test_long_sentence(self):
        self._chk("昨日公園に行って桜を見てきました。とても綺麗でした。", False)


# ================================================================
# 終了意思検出
# ================================================================
class TestDetectStopIntent(unittest.TestCase):

    def setUp(self):
        self.agent = make_agent(tempfile.mkdtemp())

    def test_sorosoro_owari(self):   self.assertTrue(self.agent.detect_stop_intent("そろそろ終わりにしましょう"))
    def test_yameru(self):           self.assertTrue(self.agent.detect_stop_intent("やめたい"))
    def test_mata_kondo(self):       self.assertTrue(self.agent.detect_stop_intent("またね、バイバイ"))
    def test_stop_keyword(self):     self.assertTrue(self.agent.detect_stop_intent("ストップ"))
    def test_owari_ni(self):         self.assertTrue(self.agent.detect_stop_intent("ここで終わりにします"))
    def test_normal_text_false(self):self.assertFalse(self.agent.detect_stop_intent("今日はいい天気ですね"))
    def test_none_false(self):       self.assertFalse(self.agent.detect_stop_intent(None))


# ================================================================
# フェーズ遷移
# ================================================================
class TestTransitionPolicy(unittest.TestCase):

    def setUp(self):
        self.agent = make_agent(tempfile.mkdtemp())

    def test_setup_to_intro(self):
        self.agent.phase = Phase.SETUP
        self.agent.transition_policy(obs(ActionType.SILENCE))
        self.assertEqual(self.agent.phase, Phase.INTRO)

    def test_intro_response_to_surroundings(self):
        self.agent.phase = Phase.INTRO
        self.agent.transition_policy(obs(ActionType.NORMAL))
        self.assertEqual(self.agent.phase, Phase.SURROUNDINGS)

    def test_intro_silence_stays(self):
        self.agent.phase = Phase.INTRO
        self.agent.transition_policy(obs(ActionType.SILENCE))
        self.assertEqual(self.agent.phase, Phase.INTRO)

    def test_surroundings_to_bridge(self):
        self.agent.phase = Phase.SURROUNDINGS
        self.agent.turn_in_phase = 1
        self.agent.transition_policy(obs(ActionType.NORMAL))
        self.assertEqual(self.agent.phase, Phase.BRIDGE)

    def test_bridge_memory_to_deep_dive(self):
        self.agent.phase = Phase.BRIDGE
        self.agent.transition_policy(obs(ActionType.NORMAL, memory_flag=True))
        self.assertEqual(self.agent.phase, Phase.DEEP_DIVE)

    def test_bridge_disclosure_to_deep_dive(self):
        self.agent.phase = Phase.BRIDGE
        self.agent.transition_policy(obs(ActionType.DISCLOSURE))
        self.assertEqual(self.agent.phase, Phase.DEEP_DIVE)

    def test_bridge_fail_returns_to_surroundings(self):
        self.agent.phase = Phase.BRIDGE
        self.agent.bridge_fail_count = 1
        self.agent.p_want_talk = 0.3
        self.agent.transition_policy(obs(ActionType.SILENCE))
        self.assertEqual(self.agent.phase, Phase.SURROUNDINGS)

    def test_double_silence_low_p_to_ending(self):
        self.agent.phase = Phase.BRIDGE
        self.agent.consecutive_silence = 1
        self.agent.p_want_talk = 0.10
        self.agent.transition_policy(obs(ActionType.SILENCE))
        self.assertEqual(self.agent.phase, Phase.ENDING)

    def test_double_silence_high_p_to_surroundings(self):
        self.agent.phase = Phase.BRIDGE
        self.agent.consecutive_silence = 1
        self.agent.p_want_talk = 0.5
        self.agent.transition_policy(obs(ActionType.SILENCE))
        self.assertEqual(self.agent.phase, Phase.SURROUNDINGS)

    def test_deep_dive_drop_low_p_to_ending(self):
        self.agent.phase = Phase.DEEP_DIVE
        self.agent.deep_drop_count = 1
        self.agent.p_want_talk = 0.20
        self.agent.transition_policy(obs(ActionType.NORMAL))
        self.assertEqual(self.agent.phase, Phase.ENDING)

    def test_deep_dive_drop_high_p_to_surroundings(self):
        self.agent.phase = Phase.DEEP_DIVE
        self.agent.deep_drop_count = 1
        self.agent.p_want_talk = 0.40
        self.agent.transition_policy(obs(ActionType.NORMAL))
        self.assertEqual(self.agent.phase, Phase.SURROUNDINGS)

    def test_speech_resets_consecutive_silence(self):
        self.agent.phase = Phase.SURROUNDINGS
        self.agent.consecutive_silence = 3
        self.agent.turn_in_phase = 0
        self.agent.transition_policy(obs(ActionType.NORMAL))
        self.assertEqual(self.agent.consecutive_silence, 0)

    def test_ending_stays(self):
        self.agent.phase = Phase.ENDING
        self.agent.transition_policy(obs(ActionType.SILENCE))
        self.assertEqual(self.agent.phase, Phase.ENDING)


# ================================================================
# 会話ログ パース
# ================================================================
class TestLoadHistoryAsMessages(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.agent = make_agent(self.tmp)

    def _write(self, content, name="log_test.txt"):
        path = os.path.join(self.tmp, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        self.agent.history_file = path

    def test_parse_roles(self):
        self._write(
            "[10:00:00] AI: こんにちは。\n"
            "[10:00:05] User: まあまあです。\n"
            "[10:00:10] AI: それは良かった。\n"
        )
        msgs = self.agent.load_history_as_messages(max_messages=10)
        self.assertEqual(len(msgs), 3)
        self.assertEqual(msgs[0]["role"], "assistant")
        self.assertIn("こんにちは", msgs[0]["content"])
        self.assertEqual(msgs[1]["role"], "user")

    def test_max_messages_limit(self):
        lines = "".join(
            f"[10:00:{i:02d}] {'AI' if i % 2 == 0 else 'User'}: msg{i}\n"
            for i in range(20)
        )
        self._write(lines, "log_many.txt")
        self.assertLessEqual(len(self.agent.load_history_as_messages(max_messages=6)), 6)

    def test_empty_file(self):
        self._write("")
        self.assertEqual(self.agent.load_history_as_messages(), [])

    def test_nonexistent_file(self):
        self.agent.history_file = os.path.join(self.tmp, "nope.txt")
        self.assertEqual(self.agent.load_history_as_messages(), [])


# ================================================================
# 直近質問抽出
# ================================================================
class TestExtractRecentAssistantQuestions(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.agent = make_agent(self.tmp)

    def _write(self, content):
        path = os.path.join(self.tmp, "log_q.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        self.agent.history_file = path

    def test_extracts_question_marks(self):
        self._write(
            "[10:00:00] AI: 今日はどんな気分ですか？\n"
            "[10:00:05] User: まあまあ。\n"
            "[10:00:10] AI: 公園に行きましたか？\n"
        )
        qs = self.agent._extract_recent_assistant_questions()
        self.assertGreater(len(qs), 0)
        self.assertTrue(any("？" in q or "ですか" in q for q in qs))

    def test_no_question_sentences(self):
        self._write(
            "[10:00:00] AI: そうですね。\n"
            "[10:00:05] User: はい。\n"
        )
        qs = self.agent._extract_recent_assistant_questions()
        self.assertIsInstance(qs, list)


if __name__ == "__main__":
    unittest.main(verbosity=2)
