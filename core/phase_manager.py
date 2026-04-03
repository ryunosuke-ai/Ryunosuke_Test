"""フェーズ遷移ポリシー + 設定テーブル + インタラクションモード指示"""

import logging
from typing import Dict

from core.models import Phase, PhaseConfig, Observation, ActionType


# フェーズ設定テーブル（デフォルト）
DEFAULT_PHASE_CONFIGS: Dict[Phase, PhaseConfig] = {
    Phase.SETUP: PhaseConfig(
        name=Phase.SETUP,
        require_image=False,
        max_turns=1,
        instruction=(
            "目的共有（超短く）：\n"
            "・これから写真を一緒に見ながら、思い出話につながる会話をします\n"
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
            "・確認は答えやすく自然な質問にする：（例：「今日はどんな1日を過ごしていますか？」や「最近何か変わったことはありましたか？」など）\n"
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
            "・まずは直前のユーザーの返答に対して、返答内容に沿った感想・共感を簡単に述べること\n"
            "・その後、画像についての話題に移る際には必ず接続詞（「ところで」「そういえば」など）を入れてから画像の話に移る\n"
            "・ユーザーが入りやすい余白を作る（間、共感、短い確認）\n"
            "・画像の中から会話が盛り上がりそうな要素を取り出して質問を考えてください\n"
            "・質問するなら負担小：はい/いいえ、二択、指差しレベルの1問だけ\n"
            "・会話メモにある内容は繰り返し聞かず、言及する形でつなぐ\n"
            "・反応が薄いなら、質問を減らしてコメント中心にする\n"
        ),
    ),
    Phase.BRIDGE: PhaseConfig(
        name=Phase.BRIDGE,
        require_image=True,
        max_turns=4,
        instruction=(
            "連想。\n"
            "・直前のユーザーの返答に対して、返答内容に沿った共感やリアクションを示してから次の話題へ繋ぐこと\n"
            "・画像内の要素→連想の足場→過去/好みへ\u201c自然に\u201dつなぐ\n"
            "・足場例：場所→季節→行事→食べ物→人（いきなり深い話は聞かない）\n"
            "・回想や積極的な語りが出たら深掘りへ。反応が薄ければ周囲共有に戻ってよい\n"
            "・質問は最大1つ。連続質問は禁止。相手の負担が高そうなら質問を減らす\n"
            "・【重要】会話メモ（すでに確認済み）にユーザーの好み（例：コーヒーが好き等）が書かれている場合、その前提で話を進めること。絶対に同じ質問を再度ゼロから聞かない。\n"
        ),
    ),
    Phase.DEEP_DIVE: PhaseConfig(
        name=Phase.DEEP_DIVE,
        require_image=False,
        max_turns=6,
        instruction=(
            "深掘り。\n"
            "・【重要】基本は 共感・リアクション → 要約 → （必要なら）質問1つ。質問より理解と受容を優先する\n"
            "・深める：感情/意味（例：その時どんなお気持ちでしたか？）\n"
            "・広げる：関連記憶（例：似た体験は他にもありましたか？）\n"
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
            "・【重要】ユーザーの最後の言葉にしっかり共感・リアクションしてから締める\n"
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
        self.consecutive_empathy_only: int = 0

    def transition_policy(self, obs: Observation, p_want_talk: float) -> None:
        """フェーズ遷移の核。p_want_talk は外部から受け取る。"""
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
            self._set_phase(Phase.SURROUNDINGS, reason="導入完了")
            return

        if self.phase == Phase.SURROUNDINGS:
            if self.turn_in_phase >= 1:
                self._set_phase(Phase.BRIDGE, reason="共同注意ができたので連想へ")
            return

        if self.phase == Phase.BRIDGE:
            if obs.action_type == ActionType.DISENGAGE:
                self.bridge_fail_count = 0
                if p_want_talk < 0.30:
                    self._set_phase(Phase.ENDING, reason="連想で拒否が出たため終了へ")
                else:
                    self._set_phase(Phase.SURROUNDINGS, reason="連想で負担が高そうなので周囲共有へ戻す")
                return
            if obs.memory_flag or obs.action_type == ActionType.ACTIVE:
                self.bridge_fail_count = 0
                self._set_phase(Phase.DEEP_DIVE, reason="回想または積極的な語りが出たため深掘りへ")
                return
            if obs.action_type == ActionType.MINIMAL or p_want_talk < 0.35:
                self.bridge_fail_count += 1
            else:
                self.bridge_fail_count = 0
            if self.bridge_fail_count >= 2:
                self.bridge_fail_count = 0
                self._set_phase(Phase.SURROUNDINGS, reason="深まりにくいので周囲共有に戻す")
                return
            return

        if self.phase == Phase.DEEP_DIVE:
            if obs.action_type == ActionType.DISENGAGE:
                self.deep_drop_count = 0
                if p_want_talk < 0.30:
                    self._set_phase(Phase.ENDING, reason="深掘りで拒否が出たため終了へ")
                else:
                    self._set_phase(Phase.SURROUNDINGS, reason="深掘りで拒否が出たため周囲共有へ戻す")
                return
            if obs.action_type in [ActionType.ACTIVE, ActionType.RESPONSIVE]:
                self.deep_drop_count = 0
                return
            if obs.action_type == ActionType.MINIMAL:
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

    def notify_reply(self, reply_text: str) -> None:
        """AIの応答に質問が含まれているかを記録する。"""
        if not reply_text or ("？" not in reply_text and "?" not in reply_text):
            self.consecutive_empathy_only += 1
        else:
            self.consecutive_empathy_only = 0

    def get_interaction_mode_instruction(self, obs: Observation, p_want_talk: float) -> str:
        """質問をしすぎないためのモード指示を返す。"""
        if self.phase == Phase.ENDING:
            return "【モード】終了：質問禁止。感想と感謝で閉じる。"

        if obs.action_type == ActionType.DISENGAGE:
            return (
                "【モード】負荷を下げる。\n"
                "・まず拒否や負担感を受け止める\n"
                "・新しい深掘り質問はしない\n"
                "・必要なら話題を戻すか、自然に締めへ向かう"
            )

        if obs.action_type == ActionType.MINIMAL:
            return (
                "【モード】軽い誘導。\n"
                "・短いコメント→負担の小さい質問を1つ（はい/いいえ or 二択）\n"
                "・同じことは聞き返さない"
            )

        if obs.action_type == ActionType.ACTIVE or p_want_talk >= 0.70:
            if self.consecutive_empathy_only >= 1:
                return (
                    "【モード】共感＋質問。\n"
                    "・前回は共感のみだったので、今回は必ず質問を1つ含める\n"
                    "・共感/要約を述べた後、話を広げる質問を1つ添える"
                )
            return (
                "【モード】共感重視。\n"
                "・ユーザが話したい気持ちが強いため、ユーザの返答を基に共感や感想を中心にする\n"
                "・質問は最大1つ。ただし、共感重視なので、質問はしなくても良い。"
            )
        if obs.action_type == ActionType.RESPONSIVE or p_want_talk >= 0.40:
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
            self.consecutive_empathy_only = 0
