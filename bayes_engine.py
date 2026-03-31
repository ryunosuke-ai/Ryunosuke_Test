"""ベイズ更新 + 観測分類（LLM利用）"""

import json
import logging
import re
from typing import Dict, Optional, Tuple

from models import ActionType, ClassificationResult


# --- LLMプロンプト定数 ---

CLASSIFY_ACTION_PROMPT = (
    "あなたは対話分析の専門家です。ユーザー発話を、会話継続への姿勢に基づいて "
    "ACTIVE / RESPONSIVE / MINIMAL / DISENGAGE の4値のいずれか1つに分類してください。\n"
    "\n"
    "## 分類原則\n"
    "- 長さではなく、会話を続ける方向に働いているかを優先する\n"
    "- 感情の強さではなく、会話参加の姿勢をみる\n"
    "- 必ず4値のどれか1つに落とす\n"
    "\n"
    "## ラベル定義\n"
    "\n"
    "### ACTIVE\n"
    "- 自発的に話を広げる、追加情報を出す、話題を拡張する、相手に聞き返す\n"
    "- 例: 『高校の頃によくその公園に行ってたんです。夏祭りが好きでした』\n"
    "\n"
    "### RESPONSIVE\n"
    "- 問いに対して意味のある返答をしているが、自発的な拡張は強くない\n"
    "- 例: 『はい、行ったことがあります』『学生の頃です』\n"
    "\n"
    "### MINIMAL\n"
    "- 最小限の反応はあるが、内容の広がりが乏しく、積極性が弱い\n"
    "- 例: 『うん』『そうですね』『まあ』『たぶん』\n"
    "\n"
    "### DISENGAGE\n"
    "- 会話継続に消極的で、拒否・終了・回避の方向が明確\n"
    "- 例: 『その話はしたくないです』『今日はこのへんで』\n"
    "\n"
    "## 境界ルール\n"
    "- 短くても質問への意味ある返答なら RESPONSIVE\n"
    "- 詳しくても拒否や終了意図が中心なら DISENGAGE\n"
    "- 単なる相づちや曖昧応答は MINIMAL\n"
    "- 必要最小限を超えて話を前に進めるなら ACTIVE\n"
    "\n"
    "出力は JSON のみ: "
    "{\"primary_label\": \"ACTIVE|RESPONSIVE|MINIMAL|DISENGAGE\", "
    "\"reason\": \"判定理由を1〜3文で簡潔に\"}"
)

JUDGE_MEMORY_SIGNAL_PROMPT = (
    "あなたは対話分析の専門家です。ユーザー発話に、回想や具体的なエピソード記憶が含まれるかを判定してください。\n"
    "\n"
    "### memory_flag を true にする条件\n"
    "- 過去の出来事を時間的に位置づけて語っている\n"
    "- 特定の場面・場所・人物が登場する体験を述べている\n"
    "- 個人的な出来事を過去形で語っている\n"
    "\n"
    "### false にする条件\n"
    "- 現在の好みや評価だけを述べている\n"
    "- 相づちや短い返答のみ\n"
    "- 一般論や客観的事実のみ\n"
    "- 習慣の説明だけで、特定の出来事を指していない\n"
    "\n"
    "出力は JSON のみ: "
    "{\"memory_flag\": true/false, \"note\": \"短い所見（任意）\"}"
)


# 尤度テーブル（デフォルト値）
DEFAULT_LIKELIHOODS: Dict[str, Dict[ActionType, float]] = {
    "H1": {
        ActionType.ACTIVE: 0.45,
        ActionType.RESPONSIVE: 0.30,
        ActionType.MINIMAL: 0.17,
        ActionType.DISENGAGE: 0.08,
    },
    "H0": {
        ActionType.ACTIVE: 0.10,
        ActionType.RESPONSIVE: 0.25,
        ActionType.MINIMAL: 0.35,
        ActionType.DISENGAGE: 0.30,
    },
}


def update_posterior(
    p_want_talk: float,
    action_type: ActionType,
    likelihoods: Optional[Dict[str, Dict[ActionType, float]]] = None,
) -> float:
    """ベイズ更新を行い、事後確率を返す純関数。"""
    if likelihoods is None:
        likelihoods = DEFAULT_LIKELIHOODS

    prior = float(p_want_talk)
    l_h1 = float(likelihoods["H1"][action_type])
    l_h0 = float(likelihoods["H0"][action_type])

    evidence = (l_h1 * prior) + (l_h0 * (1.0 - prior))
    if evidence <= 1e-12:
        posterior = prior
    else:
        posterior = (l_h1 * prior) / evidence

    return max(0.001, min(0.999, posterior))


def _call_llm(client, deployment: str, messages: list, max_tokens: int, temperature: float, logger: logging.Logger):
    """LLM呼び出しの共通ヘルパー。"""
    return client.chat.completions.create(
        model=deployment,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )


def _extract_json_object(text: str) -> dict:
    """LLM出力から最初のJSONオブジェクトを抽出する。"""
    if not text:
        return {}

    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if not match:
        return {}

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def _extract_label_from_text(text: str) -> Optional[ActionType]:
    """JSON化に失敗した場合の保守的なラベル抽出。"""
    upper = (text or "").upper()
    for label in ActionType:
        if label.value in upper:
            return label
    return None


def classify_action(client, deployment: str, user_text: Optional[str], logger: logging.Logger) -> ClassificationResult:
    """ユーザー発言を4主ラベルに分類し、理由も返す。"""
    if not user_text:
        return ClassificationResult(
            action_type=ActionType.RESPONSIVE,
            reason="入力が空のため、保守的に RESPONSIVE とみなしました。",
        )

    messages = [
        {"role": "system", "content": CLASSIFY_ACTION_PROMPT},
        {"role": "user", "content": user_text},
    ]
    try:
        res = _call_llm(client, deployment, messages, max_tokens=120, temperature=0, logger=logger)
        text = (res.choices[0].message.content or "").strip()
        payload = _extract_json_object(text)

        raw_label = str(payload.get("primary_label", "")).strip().upper()
        reason = str(payload.get("reason", "")).strip() or None

        if raw_label in ActionType._value2member_map_:
            return ClassificationResult(action_type=ActionType(raw_label), reason=reason)

        fallback_label = _extract_label_from_text(text)
        if fallback_label:
            return ClassificationResult(action_type=fallback_label, reason=reason)

        logger.warning("分類LLMの出力を解釈できませんでした: %s", text)
    except Exception as e:
        logger.warning("分類LLM失敗: %s", e)

    return ClassificationResult(
        action_type=ActionType.RESPONSIVE,
        reason="分類に失敗したため、保守的に RESPONSIVE とみなしました。",
    )


def judge_memory_signal(client, deployment: str, user_text: Optional[str], logger: logging.Logger) -> Tuple[bool, Optional[str]]:
    """BRIDGE/DEEP_DIVEで使う観測: memory_flag, note。"""
    if not user_text:
        return False, None

    messages = [
        {"role": "system", "content": JUDGE_MEMORY_SIGNAL_PROMPT},
        {"role": "user", "content": user_text},
    ]
    try:
        res = _call_llm(client, deployment, messages, max_tokens=80, temperature=0, logger=logger)
        text = (res.choices[0].message.content or "").strip()
        payload = _extract_json_object(text)

        if payload:
            memory_flag = bool(payload.get("memory_flag", False))
            note = str(payload.get("note", "")).strip() or None
            return memory_flag, note

        match = re.search(r'"memory_flag"\s*:\s*(true|false)', text, re.IGNORECASE)
        note_match = re.search(r'"note"\s*:\s*"([^"]*)"', text)
        memory_flag = (match.group(1).lower() == "true") if match else False
        note = note_match.group(1) if note_match else None
        return memory_flag, note
    except Exception as e:
        logger.warning("回想判定LLM失敗: %s", e)
        return False, None


def judge_memory_and_disclosure(
    client, deployment: str, user_text: Optional[str], logger: logging.Logger
) -> Tuple[bool, bool, Optional[str]]:
    """後方互換のために残す。自己開示軸は使用しない。"""
    memory_flag, note = judge_memory_signal(client, deployment, user_text, logger)
    return memory_flag, False, note
