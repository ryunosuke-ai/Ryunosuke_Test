"""ベイズ更新 + 観測分類（LLM利用）"""

import re
import json
import logging
from typing import Optional, Tuple, Dict

from models import ActionType


# 尤度テーブル（デフォルト値）
DEFAULT_LIKELIHOODS: Dict[str, Dict[ActionType, float]] = {
    "H1": {
        ActionType.SILENCE: 0.10,
        ActionType.NORMAL: 0.25,
        ActionType.DISCLOSURE: 0.65,
    },
    "H0": {
        ActionType.SILENCE: 0.45,
        ActionType.NORMAL: 0.50,
        ActionType.DISCLOSURE: 0.05,
    },
}

# 生返事に対する尤度（NORMAL を上書き）
DEFAULT_MINIMAL_LIKELIHOOD: Dict[str, float] = {
    "H1": 0.05,
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

    prompt = (
        "次のユーザー発言を分類してください。\n"
        "・DISCLOSURE: ①過去の体験（昔〜した等）②個人の好み/価値観（好き/苦手/こだわり等）③感情（嬉しい/つらい等）"
        "④具体的なエピソード（いつ/どこで/誰と など）が含まれる。\n"
        "  ※『今日は部屋でゆっくりしてます』『特に予定ないです』のような\u201c現在の状況だけ\u201dは NORMAL。\n"
        "・NORMAL: 相づち、短い返答、現在状況の一言、事実のみで広がりにくい。\n"
        "出力は DISCLOSURE か NORMAL のどちらか1語のみ。"
    )
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

    prompt = (
        "次のユーザー発言について判定してください。\n"
        "1) memory_flag: 過去の出来事・体験として語っているなら true。\n"
        "2) self_disclosure_flag: 好み/考え/感情など自己開示が含まれるなら true。\n"
        "出力は JSON のみ: {\"memory_flag\": true/false, \"self_disclosure_flag\": true/false, \"note\": \"短い所見（任意）\"}\n"
    )
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
