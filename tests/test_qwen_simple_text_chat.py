"""Qwen3.5 テキスト会話の再生成・表示フォールバックのテスト。"""

from llm.qwen.simple_text_chat_qwen35 import SimpleTextChatAgent


class FakeTensor:
    """最低限の shape と to を持つ簡易テンソル。"""

    def __init__(self, values):
        self.values = values
        self.shape = (1, len(values))

    def to(self, device):
        return self


class FakeOutputRow:
    """生成結果のスライスを模したスタブ。"""

    def __init__(self, values):
        self.values = values

    def __getitem__(self, item):
        if isinstance(item, slice):
            return self.values[item]
        return self.values[item]


class FakeOutput:
    """generate の返り値を模したスタブ。"""

    def __init__(self, generated_tokens):
        self.generated_tokens = generated_tokens

    def __getitem__(self, index):
        assert index == 0
        return FakeOutputRow([1, 2, 3] + self.generated_tokens)


class FakeTokenizer:
    """生成結果の decode とテンプレート記録だけを行う。"""

    eos_token_id = 99

    def __init__(self, with_special_texts, without_special_texts=None):
        self.with_special_texts = list(with_special_texts)
        self.without_special_texts = list(without_special_texts or with_special_texts)
        self.template_calls = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, enable_thinking=False):
        self.template_calls.append({
            "messages": messages,
            "enable_thinking": enable_thinking,
        })
        return f"prompt-{len(self.template_calls)}"

    def __call__(self, prompt, return_tensors="pt"):
        assert return_tensors == "pt"
        return {
            "input_ids": FakeTensor([1, 2, 3]),
            "attention_mask": FakeTensor([1, 1, 1]),
        }

    def decode(self, generated_ids, skip_special_tokens=False):
        if skip_special_tokens:
            return self.without_special_texts.pop(0)
        return self.with_special_texts.pop(0)


class FakeModel:
    """決め打ちの生成結果を返す。"""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = 0

    def generate(self, **kwargs):
        output = self.outputs[self.calls]
        self.calls += 1
        return FakeOutput(output)


class FakeLogger:
    """ロガー呼び出しを記録する。"""

    def __init__(self):
        self.records = []

    def info(self, message, *args):
        self.records.append(("info", message % args if args else message))

    def warning(self, message, *args):
        self.records.append(("warning", message % args if args else message))


def _make_agent(with_special_texts, without_special_texts=None, outputs=None):
    agent = object.__new__(SimpleTextChatAgent)
    agent.enable_thinking = True
    agent.show_thinking = False
    agent.local_device = "cuda"
    agent.local_max_new_tokens = 180
    agent.local_temperature = 1.0
    agent.local_top_p = 0.95
    agent.local_top_k = 20
    agent.local_min_p = 0.0
    agent.local_repetition_penalty = 1.0
    agent.tokenizer = FakeTokenizer(with_special_texts, without_special_texts)
    output_count = len(outputs) if outputs is not None else max(len(with_special_texts), 1)
    agent.local_model = FakeModel(outputs or [[4, 5, 6]] * output_count)
    agent.logger = FakeLogger()
    agent.p_want_talk = 0.33
    agent.flow_stage = "WARMUP"
    agent.conv_memory = type("Memory", (), {"summary": "", "do_not_ask": []})()
    agent.load_history_as_messages = lambda max_messages=10: []
    agent._build_interaction_mode_instruction = lambda obs, ending_mode: "【モード】軽い再点火。\n"
    agent._build_stage_instruction = lambda: "【会話段階】WARMUP。\n"
    agent._build_stage_goal_instruction = lambda: "【段階目標】話しやすい空気を作る。"
    agent._build_reminiscence_examples = lambda: ""
    agent._delta_window = lambda: 0.0
    return agent


def test_think_and_reply_retries_when_first_output_is_reasoning_only():
    agent = _make_agent(
        with_special_texts=[
            "Thinking Process:\n\n1. Analyze.",
            "Thinking Process:\n\n1. Analyze.",
            "こんにちは。今日はゆっくりできていますか？",
            "こんにちは。今日はゆっくりできていますか？",
        ],
        without_special_texts=[
            "Thinking Process:\n\n1. Analyze.",
            "こんにちは。今日はゆっくりできていますか？",
        ],
        outputs=[[4], [5]],
    )
    obs = type("Obs", (), {"user_text": "こんにちは！", "action_type": type("Action", (), {"value": "MINIMAL"})()})()

    result = agent.think_and_reply(obs)

    assert result == "こんにちは。今日はゆっくりできていますか？"
    assert agent.local_model.calls == 2
    assert len(agent.tokenizer.template_calls) == 2
    assert all(call["enable_thinking"] is True for call in agent.tokenizer.template_calls)
    assert any("再生成" in record[1] for record in agent.logger.records if record[0] == "warning")


def test_think_and_reply_returns_raw_output_when_retry_still_has_no_final_answer():
    agent = _make_agent(
        with_special_texts=[
            "<think>Thinking Process:\n\n1. Analyze first.",
            "<think>Thinking Process:\n\n1. Analyze first.",
            "<think>Thinking Process:\n\n2. Still reasoning only.",
            "<think>Thinking Process:\n\n2. Still reasoning only.",
        ],
        without_special_texts=[
            "<think>Thinking Process:\n\n1. Analyze first.",
            "<think>Thinking Process:\n\n2. Still reasoning only.",
        ],
        outputs=[[4], [5]],
    )
    agent.local_max_new_tokens = 1
    obs = type("Obs", (), {"user_text": "こんにちは！", "action_type": type("Action", (), {"value": "MINIMAL"})()})()

    result = agent.think_and_reply(obs)

    assert result == "少し長く考えています。もう一度だけ話しかけてください。"
    assert agent.local_model.calls == 2
    assert any("max_new_tokens に到達しました" in record[1] for record in agent.logger.records if record[0] == "warning")


def test_think_and_reply_retries_with_thinking_off_when_think_block_is_truncated():
    agent = _make_agent(
        with_special_texts=[
            "<think>Thinking Process:\n\n1. Analyze first.",
            "<think>Thinking Process:\n\n1. Analyze first.",
            "こんにちは。今日はゆっくりできていますか？<|im_end|>",
            "こんにちは。今日はゆっくりできていますか？",
        ],
        without_special_texts=[
            "<think>Thinking Process:\n\n1. Analyze first.",
            "こんにちは。今日はゆっくりできていますか？",
        ],
        outputs=[[4], [5]],
    )
    agent.local_max_new_tokens = 1
    obs = type("Obs", (), {"user_text": "こんにちは！", "action_type": type("Action", (), {"value": "MINIMAL"})()})()

    result = agent.think_and_reply(obs)

    assert result == "こんにちは。今日はゆっくりできていますか？"
    assert agent.local_model.calls == 2
    assert len(agent.tokenizer.template_calls) == 2
    assert agent.tokenizer.template_calls[0]["enable_thinking"] is True
    assert agent.tokenizer.template_calls[1]["enable_thinking"] is False
    assert any("thinking を無効化" in record[1] for record in agent.logger.records if record[0] == "warning")
