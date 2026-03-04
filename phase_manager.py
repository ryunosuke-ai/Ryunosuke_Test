"""フェーズ遷移ポリシー + 設定テーブル + インタラクションモード指示"""

import logging
from typing import Dict

from models import Phase, PhaseConfig, Observation, ActionType


# フェーズ設定テーブル（デフォルト）
DEFAULT_PHASE_CONFIGS: Dict[Phase, PhaseConfig] = {
    Phase.SETUP: PhaseConfig(
        name=Phase.SETUP,
        require_image=False,
        max_turns=1,
        instruction=(
            "目的共有（超短く）：\n"
            "・これから周りを一緒に見ながら、思い出話につながる会話をします\n"
            "・答えづらい時は無理に答えなくてOK（沈黙でもOK）\n"
            "・質問は基本しない（挨拶だけで十分）\n"
            "・同じことを聞き返さないよう、会話メモを参照する\n"
        ),
    ),
    Phase.INTRO: PhaseConfig(
        name=Phase.INTRO,
        require_image=False,
        max_turns=2,
        instruction=(
            "挨拶と簡単なアイスブレイク。\n"
            "・明るく挨拶し、相手の様子を一言で受け止める\n"
            "・確認は軽く：体調/気分を1つだけ（例：今日はどんな気分？）\n"
            "・返事が短い/薄い場合は、質問を重ねずコメントで次へ\n"
            "・質問は最大1つ。連続質問は禁止\n"
            "・すでに聞いた項目（do_not_ask）は絶対に再質問しない\n"
        ),
    ),
    Phase.SURROUNDINGS: PhaseConfig(
        name=Phase.SURROUNDINGS,
        require_image=True,
        max_turns=3,
        instruction=(
            "画像についての話（共同注意）。\n"
            "・ユーザーの答えを受け止め、画像の中の要素を基に短いコメントを返す\n"
            "・ユーザーが入りやすい余白を作る（間、共感、短い確認）\n"
            "・質問するなら負担小：はい/いいえ、二択、指差しレベルの1問だけ\n"
            "・会話メモにある内容（例：今日の予定）は繰り返し聞かず、言及する形でつなぐ\n"
            "・反応が薄いなら、質問を減らしてコメント中心にする\n"
        ),
    ),
    Phase.BRIDGE: PhaseConfig(
        name=Phase.BRIDGE,
        require_image=True,
        max_turns=4,
        instruction=(
            "連想（回想の点火）。\n"
            "・周囲の要素→連想の足場→過去/好みへ\u201c自然に\u201dつなぐ\n"
            "・足場例：場所→季節→行事→食べ物→人（いきなり深い話は聞かない）\n"
            "・回想/自己開示が出たら深掘りへ。出なければ周囲共有に戻ってよい\n"
            "・質問は最大1つ。連続質問は禁止。相手の負担が高そうなら質問を減らす\n"
            "・【重要】会話メモ（すでに確認済み）にユーザーの好み（例：コーヒーが好き等）が書かれている場合、その前提で話を進めること。絶対に同じ質問（「何を飲みたいか」など）を再度ゼロから聞かないこと。\n"
        ),
    ),
    Phase.DEEP_DIVE: PhaseConfig(
        name=Phase.DEEP_DIVE,
        require_image=False,
        max_turns=6,
        instruction=(
            "深掘り。\n"
            "・基本は 共感→要約→（必要なら）質問1つ。質問より理解を優先\n"
            "・深める：感情/意味（例：その時どんな気持ち？）\n"
            "・広げる：関連記憶（例：似た体験は他にも？）\n"
            "・整理：短い要約を返して確認（例：つまり〜ということですね）\n"
            "・反応が落ちたら、深掘りをやめて周囲共有へ戻す/終盤へ移る\n"
            "・過去に答えた質問を言い換えて再質問しない\n"
        ),
    ),
    Phase.ENDING: PhaseConfig(
        name=Phase.ENDING,
        require_image=False,
        max_turns=2,
        instruction=(
            "エンディング（必ず終える）。\n"
            "・ユーザーが終えたいと言ったら、最優先で丁寧に締める\n"
            "・感謝→短いまとめ→終了の挨拶\n"
            "・質問は禁止。新しい話題を出さない\n"
            "・相手が追加で話し出しても、短く受け止めて締め直す\n"
        ),
    ),
}


# フェーズ順序
_PHASE_ORDER = [Phase.SETUP, Phase.INTRO, Phase.SURROUNDINGS, Phase.BRIDGE, Phase.DEEP_DIVE, Phase.ENDING]


class PhaseManager:
    """フェーズ状態を保持し、遷移ポリシーを適用する。"""

    def __init__(self, phase_configs: Dict[Phase, PhaseConfig] = None, logger: logging.Logger = None):
        self.phase_configs = phase_configs or DEFAULT_PHASE_CONFIGS
        self.logger = logger or logging.getLogger("phase_manager")

        self.phase: Phase = Phase.SETUP
        self.turn_in_phase: int = 0
        self.consecutive_silence: int = 0
        self.bridge_fail_count: int = 0
        self.deep_drop_count: int = 0

    def transition_policy(self, obs: Observation, p_want_talk: float) -> None:
        """フェーズ遷移の核。p_want_talk は外部から受け取る。"""
        # 沈黙カウント
        if obs.action_type == ActionType.SILENCE:
            self.consecutive_silence += 1
        else:
            self.consecutive_silence = 0

        # 連続沈黙が続くなら負荷を下げる
        if self.consecutive_silence >= 2 and self.phase not in [Phase.ENDING]:
            if p_want_talk < 0.20:
                self._set_phase(Phase.ENDING, reason="沈黙が続き、負担が高そう")
                return
            self._set_phase(Phase.SURROUNDINGS, reason="沈黙が続いたため、周囲共有へ戻す")
            return

        cfg = self.phase_configs[self.phase]

        # フェーズ内ターン数上限
        if self.turn_in_phase >= cfg.max_turns and self.phase not in [Phase.BRIDGE, Phase.DEEP_DIVE]:
            nxt = self._next_phase_linear(self.phase)
            self._set_phase(nxt, reason="フェーズ上限ターン到達")
            return

        # 個別ルール
        if self.phase == Phase.SETUP:
            self._set_phase(Phase.INTRO, reason="セット完了")
            return

        if self.phase == Phase.INTRO:
            if obs.action_type != ActionType.SILENCE:
                self._set_phase(Phase.SURROUNDINGS, reason="導入完了")
            return

        if self.phase == Phase.SURROUNDINGS:
            if self.turn_in_phase >= 1 and obs.action_type != ActionType.SILENCE:
                self._set_phase(Phase.BRIDGE, reason="共同注意ができたので連想へ")
            return

        if self.phase == Phase.BRIDGE:
            if obs.memory_flag or obs.action_type == ActionType.DISCLOSURE:
                self.bridge_fail_count = 0
                self._set_phase(Phase.DEEP_DIVE, reason="回想が出たため深掘りへ")
                return
            if obs.action_type == ActionType.SILENCE or p_want_talk < 0.35:
                self.bridge_fail_count += 1
            if self.bridge_fail_count >= 2:
                self.bridge_fail_count = 0
                self._set_phase(Phase.SURROUNDINGS, reason="回想が出にくいので周囲共有に戻す")
                return
            return

        if self.phase == Phase.DEEP_DIVE:
            if obs.action_type == ActionType.DISCLOSURE:
                self.deep_drop_count = 0
                return
            if obs.action_type in [ActionType.NORMAL, ActionType.SILENCE]:
                self.deep_drop_count += 1
            if self.deep_drop_count >= 2:
                if p_want_talk < 0.30:
                    self._set_phase(Phase.ENDING, reason="深掘りで反応が落ちたため終了へ")
                else:
                    self._set_phase(Phase.SURROUNDINGS, reason="深掘りで反応が落ちたため周囲共有へ戻す")
                self.deep_drop_count = 0
            return

        # ENDING は維持
        return

    def get_interaction_mode_instruction(self, obs: Observation, p_want_talk: float) -> str:
        """質問をしすぎないためのモード指示を返す。"""
        if self.phase == Phase.ENDING:
            return "【モード】終了：質問禁止。感想と感謝で閉じる。"

        if obs.action_type == ActionType.SILENCE:
            return (
                "【モード】低負荷：相手を急かさない。\n"
                "・短い気遣い＋沈黙の余白\n"
                "・質問はしない（しても『大丈夫？』の1回まで）"
            )

        if obs.action_type == ActionType.NORMAL and obs.minimal_reply:
            return (
                "【モード】軽い誘導（生返事）。\n"
                "・短いコメント→負担の小さい質問を1つ（はい/いいえ or 二択）\n"
                "・同じことは聞き返さない"
            )

        if p_want_talk >= 0.70:
            return (
                "【モード】共感重視。\n"
                "・共感/要約/感想を中心\n"
                "・質問は最大1つ（できればしない）"
            )
        if p_want_talk >= 0.40:
            return (
                "【モード】軽い誘導。\n"
                "・コメント→小さな質問1つ（はい/いいえ、二択など）\n"
                "・押し付けない"
            )
        return (
            "【モード】再点火（負担は小さく）。\n"
            "・周囲要素を1つ拾って短いコメント\n"
            "・質問するなら『二択』か『はい/いいえ』にする\n"
            "・連続質問は禁止"
        )

    def _next_phase_linear(self, phase: Phase) -> Phase:
        i = _PHASE_ORDER.index(phase)
        return _PHASE_ORDER[min(i + 1, len(_PHASE_ORDER) - 1)]

    def _set_phase(self, phase: Phase, reason: str = "") -> None:
        if self.phase != phase:
            self.logger.info("⏩ フェーズ遷移: %s → %s（%s）", self.phase.value, phase.value, reason)
            self.phase = phase
            self.turn_in_phase = 0
