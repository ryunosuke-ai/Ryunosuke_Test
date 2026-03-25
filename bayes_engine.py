"""ベイズ更新 + 観測分類（LLM利用）"""

import re
import json
import logging
from typing import Optional, Tuple, Dict

from models import ActionType


# --- LLMプロンプト定数 ---

# classify_action() 用: 社会的浸透理論に基づく発話分類プロンプト
CLASSIFY_ACTION_PROMPT = (
    "あなたは対話分析の専門家です。社会的浸透理論に基づき、"
    "ユーザー発言の「自己開示の深さ」を判定してください。\n"
    "\n"
    "## 判定基準\n"
    "\n"
    "自己開示とは「自分自身に関する個人的な情報を他者に意図的に伝える行為」です。\n"
    "以下の基準で DISCLOSURE か NORMAL かを判定してください。\n"
    "\n"
    "### DISCLOSURE（自己開示あり）と判定する条件\n"
    "次のいずれか1つ以上に該当する場合：\n"
    "\n"
    "【感情・内面の表出】\n"
    "  - 感情や気持ちを表現している（嬉しい、寂しい、不安、楽しかった、つらかった 等）\n"
    "  - 価値判断や個人的な評価を含む（好き、嫌い、苦手、大事にしている 等）\n"
    "\n"
    "【個人的な体験・記憶】\n"
    "  - 過去のエピソードを語っている（昔〜した、子どもの頃〜、あの時〜 等）\n"
    "  - 具体的な場面や人物が登場する体験談（いつ・どこで・誰と・何をした）\n"
    "\n"
    "【個人の属性・背景の能動的な共有】\n"
    "  - 自分の趣味・習慣・こだわりを説明している（毎朝〜する、いつも〜している 等）\n"
    "  - 家族構成、出身地、職歴など個人的な背景情報を自ら語っている\n"
    "\n"
    "【願望・意図・将来の計画】\n"
    "  - 個人的な希望や夢を語っている（〜したい、〜に行ってみたい 等）\n"
    "  - 自分なりの考えや信念を述べている\n"
    "\n"
    "### NORMAL（自己開示なし）と判定する条件\n"
    "次のいずれかに該当する場合：\n"
    "\n"
    "【相づち・最小応答】\n"
    "  - 「はい」「うん」「そうですね」「なるほど」「ありがとう」など\n"
    "  - 会話を継続する意思表示のみで、新しい個人情報を含まない\n"
    "\n"
    "【現在の状況の事実報告のみ】\n"
    "  - 「今日は家にいます」「特に予定ないです」「天気がいいですね」など\n"
    "  - 自分の状態を述べているが、感情・評価・背景情報を伴わない単なる状況描写\n"
    "\n"
    "【一般的な知識・客観的事実】\n"
    "  - 「桜は春に咲きますね」「東京は暑いですね」など\n"
    "  - 個人に紐づかない一般論や客観的事実のみ\n"
    "\n"
    "【質問・確認】\n"
    "  - 「それはどういう意味ですか？」「何時ですか？」など\n"
    "  - 相手への質問で、自分の情報を含まない\n"
    "\n"
    "## 判定のポイント\n"
    "- 迷ったら「この発言から話者の人となり（性格・経験・価値観）が新しくわかるか？」を基準にする\n"
    "- 同じ話題でも深さが異なる場合がある：\n"
    "  ・「コーヒー飲みます」→ NORMAL（状況報告）\n"
    "  ・「コーヒーが好きなんです、毎朝必ず飲みます」→ DISCLOSURE（好み＋習慣）\n"
    "  ・「昔、祖父とよく喫茶店に行ったんです」→ DISCLOSURE（体験＋人物）\n"
    "\n"
    "出力は DISCLOSURE か NORMAL のどちらか1語のみ。"
)

# judge_memory_and_disclosure() 用: 回想・自己開示の2軸判定プロンプト
JUDGE_MEMORY_PROMPT = (
    "あなたは対話分析の専門家です。ユーザー発言について2つの軸で判定してください。\n"
    "\n"
    "## 判定軸\n"
    "\n"
    "### 1. memory_flag（回想・エピソード記憶）\n"
    "以下に該当すれば true:\n"
    "- 過去の出来事を時間的に位置づけて語っている（「昔」「子どもの頃」「あの時」「〜年前」等）\n"
    "- 特定の場面・場所・人物が登場する体験を述べている\n"
    "- 「〜したことがある」「〜だった」など過去形で個人的体験を語っている\n"
    "\n"
    "以下は false:\n"
    "- 一般的な知識や習慣の説明（「日本では〜する」）\n"
    "- 現在の状態のみ（「今日は暇です」）\n"
    "- 習慣的な行動で特定の過去の場面を指さない（「毎日散歩します」）\n"
    "\n"
    "### 2. self_disclosure_flag（自己開示）\n"
    "以下に該当すれば true:\n"
    "- 感情・気持ちを表現している（嬉しかった、寂しい、好き、嫌い 等）\n"
    "- 個人的な好み・価値観・信念を述べている（大事にしている、こだわり 等）\n"
    "- 自分の性格や特徴に言及している（人見知りで〜、心配性なので〜 等）\n"
    "- 個人的な願望や夢を語っている（〜したい、〜に行ってみたい 等）\n"
    "\n"
    "以下は false:\n"
    "- 事実のみの応答で感情・評価を含まない\n"
    "- 相づちや短い肯定のみ\n"
    "- 一般論や客観的事実の述べ立てのみ\n"
    "\n"
    "## 判定のポイント\n"
    "- memory_flag と self_disclosure_flag は独立した軸である。両方 true もありえる\n"
    "  例: 「子どもの頃、祖母の作る煮物が大好きだった」→ memory=true（過去の体験）, disclosure=true（好みの表明）\n"
    "- memory_flag のみ true の例: 「去年、東京に行きました」→ 過去の事実だが感情や好みを含まない\n"
    "- self_disclosure_flag のみ true の例: 「甘いものが好きです」→ 現在の好みで過去の体験ではない\n"
    "\n"
    "出力は JSON のみ: {\"memory_flag\": true/false, \"self_disclosure_flag\": true/false, \"note\": \"短い所見（任意）\"}"
)


# 尤度テーブル（デフォルト値）
DEFAULT_LIKELIHOODS: Dict[str, Dict[ActionType, float]] = {
    "H1": {
        ActionType.SILENCE: 0.20,
        ActionType.NORMAL: 0.30,
        ActionType.DISCLOSURE: 0.50,
    },
    "H0": {
        ActionType.SILENCE: 0.35,
        ActionType.NORMAL: 0.45,
        ActionType.DISCLOSURE: 0.20,
    },
}

# 生返事に対する尤度（NORMAL を上書き）
DEFAULT_MINIMAL_LIKELIHOOD: Dict[str, float] = {
    "H1": 0.15,
    "H0": 0.70,
}


def update_posterior(
    p_want_talk: float,
    action_type: ActionType,
    minimal_reply: bool = False,
    likelihoods: Optional[Dict] = None,
    minimal_likelihood: Optional[Dict] = None,
) -> float:
    """ベイズ更新を行い、事後確率を返す純関数。"""
    if likelihoods is None:
        likelihoods = DEFAULT_LIKELIHOODS
    if minimal_likelihood is None:
        minimal_likelihood = DEFAULT_MINIMAL_LIKELIHOOD

    prior = float(p_want_talk)

    l_h1 = float(likelihoods["H1"][action_type])
    l_h0 = float(likelihoods["H0"][action_type])

    if action_type == ActionType.NORMAL and minimal_reply:
        l_h1 = float(minimal_likelihood["H1"])
        l_h0 = float(minimal_likelihood["H0"])

    evidence = (l_h1 * prior) + (l_h0 * (1.0 - prior))
    if evidence <= 1e-12:
        posterior = prior
    else:
        posterior = (l_h1 * prior) / evidence

    return max(0.001, min(0.999, posterior))


def is_minimal_reply(user_text: Optional[str]) -> bool:
    """生返事（会話が広がりにくい最小応答）かどうか。"""
    if not user_text:
        return False
    t = user_text.strip()
    t2 = re.sub(r"[\s。．、,！!？?…]+", "", t)
    if len(t2) <= 10 and re.fullmatch(
        r"(はい|うん|そう|そうです|そうですね|なるほど|ありがとう|ありがとうございます|ええ|うーん)", t2
    ):
        return True
    if len(t2) <= 4 and any(w == t2 for w in ["はい", "うん", "そう", "ええ"]):
        return True
    return False


def _call_llm(client, deployment: str, messages: list, max_tokens: int, temperature: float, logger: logging.Logger):
    """LLM呼び出しの共通ヘルパー。"""
    return client.chat.completions.create(
        model=deployment,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )


def classify_action(client, deployment: str, user_text: Optional[str], logger: logging.Logger) -> ActionType:
    """ユーザー発言をActionTypeに分類する。"""
    if user_text is None:
        return ActionType.SILENCE

    short = len(user_text) <= 8
    if short and any(w in user_text for w in ["はい", "うん", "そう", "ありがとう", "ええ", "なるほど"]):
        return ActionType.NORMAL

    prompt = CLASSIFY_ACTION_PROMPT
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_text},
    ]
    try:
        res = _call_llm(client, deployment, messages, max_tokens=6, temperature=0, logger=logger)
        out = (res.choices[0].message.content or "").strip().upper()
        if "DISCLOSURE" in out:
            return ActionType.DISCLOSURE
        return ActionType.NORMAL
    except Exception as e:
        logger.warning("分類LLM失敗: %s", e)
        return ActionType.NORMAL


def judge_memory_and_disclosure(
    client, deployment: str, user_text: Optional[str], logger: logging.Logger
) -> Tuple[bool, bool, Optional[str]]:
    """BRIDGE/DEEP_DIVEで使う観測: memory_flag, self_disclosure_flag, note。"""
    if not user_text:
        return False, False, None

    prompt = JUDGE_MEMORY_PROMPT
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_text},
    ]
    try:
        res = _call_llm(client, deployment, messages, max_tokens=80, temperature=0, logger=logger)
        text = (res.choices[0].message.content or "").strip()
        m1 = re.search(r'"memory_flag"\s*:\s*(true|false)', text, re.IGNORECASE)
        m2 = re.search(r'"self_disclosure_flag"\s*:\s*(true|false)', text, re.IGNORECASE)
        note = None
        m3 = re.search(r'"note"\s*:\s*"([^"]*)"', text)
        if m3:
            note = m3.group(1)
        mem = (m1.group(1).lower() == "true") if m1 else False
        dis = (m2.group(1).lower() == "true") if m2 else False
        return mem, dis, note
    except Exception as e:
        logger.warning("回想判定LLM失敗: %s", e)
        return False, False, None
