"""会話メモ更新 + 終了意思検出"""

import re
import json
import logging
from typing import Optional, List

from models import MemoryUpdate


# 終了意思の正規表現パターン
_STOP_PATTERNS = [
    r"(そろそろ|もう|一旦).{0,6}(終わり|終える|やめ|切り|終了)",
    r"(会話|話).{0,6}(やめ|終わり|終了)",
    r"(終わりに|終わりで|ここまで|ストップ|停止)",
    r"(また今度|またね|ばいばい|バイバイ)",
]


def detect_stop_intent(user_text: Optional[str]) -> bool:
    """ユーザー発言から終了意思を検出する純関数。"""
    if not user_text:
        return False
    t = user_text.strip()
    for pat in _STOP_PATTERNS:
        if re.search(pat, t):
            return True
    if "やめたい" in t or "終わりたい" in t or "切りたい" in t:
        return True
    return False


def extract_recent_assistant_questions(messages: List[dict], max_questions: int = 8) -> List[str]:
    """直近のAI発話から質問文を抽出する純関数。"""
    qs: List[str] = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        text = str(msg.get("content", ""))
        candidates = re.split(r"[。\n]", text)
        for c in candidates:
            c = c.strip()
            if not c:
                continue
            if ("？" in c) or ("ですか" in c) or ("でしょうか" in c) or c.endswith("か"):
                if len(c) > 80:
                    c = c[:80] + "…"
                qs.append(c)
    seen = set()
    uniq = []
    for q in qs:
        if q in seen:
            continue
        seen.add(q)
        uniq.append(q)
    return uniq[-max_questions:]


def update_conv_memory(
    client,
    deployment: str,
    conv_memory: MemoryUpdate,
    user_text: Optional[str],
    history: List[dict],
    recent_questions: List[str],
    logger: logging.Logger,
) -> MemoryUpdate:
    """会話メモを更新し、新しいMemoryUpdateを返す。"""
    if not user_text:
        return conv_memory

    stop_intent_rb = detect_stop_intent(user_text)

    prompt = (
        "あなたは会話ログを整理するアシスタントです。\n"
        "入力：既存の要約、直近の会話履歴、直近のAIの質問一覧、ユーザー発言。\n"
        "出力：JSONのみ。\n"
        "必ず次のキーを含める：\n"
        "  summary: これまでの会話の要約（日本語、500文字以内、簡潔に）\n"
        "  do_not_ask: すでにユーザーが答えた/確定した内容で、AIが繰り返し質問すべきでない『具体的な項目』の短いリスト（最大8件）。\n"
        "              ※「好きな飲み物（コーヒー）」「好きな場所（静かなカフェ）」のように、ジャンルと回答内容をセットで書くこと。\n"
        "              ※過去の要約と重複しないようにし、絶対に同じことをAIが聞かないように防御するためのリストです。\n"
        "  stop_intent: ユーザーが会話を終えたい意思があるなら true（今回の発言から判断）\n"
        "\n"
    )

    payload = {
        "current_summary": conv_memory.summary,
        "current_do_not_ask": conv_memory.do_not_ask,
        "recent_questions": recent_questions,
        "history": history[-12:],
        "user_text": user_text,
    }

    try:
        res = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            max_tokens=512,
            temperature=0,
        )
        text = (res.choices[0].message.content or "").strip()
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            raise ValueError("JSONが見つかりません")
        data = json.loads(m.group(0))

        summary = str(data.get("summary", "")).strip()
        dna = data.get("do_not_ask", [])
        if not isinstance(dna, list):
            dna = []
        dna = [str(x).strip() for x in dna if str(x).strip()]
        dna = dna[:8]

        si = bool(data.get("stop_intent", False)) or stop_intent_rb

        return MemoryUpdate(summary=summary, do_not_ask=dna, stop_intent=si)

    except Exception as e:
        logger.warning("会話メモ更新失敗: %s", e)
        # ルールベースのフォールバック
        new_memory = MemoryUpdate(
            summary=conv_memory.summary,
            do_not_ask=list(conv_memory.do_not_ask),
            stop_intent=stop_intent_rb or conv_memory.stop_intent,
        )
        if any(k in user_text for k in ["今日", "予定", "過ご", "する"]):
            if "今日の予定" not in new_memory.do_not_ask:
                new_memory.do_not_ask.append("今日の予定")
        if any(k in user_text for k in ["好き", "好み", "ジャンル", "作家", "作品"]):
            if "好み（ジャンル/作家/作品）" not in new_memory.do_not_ask:
                new_memory.do_not_ask.append("好み（ジャンル/作家/作品）")
        new_memory.do_not_ask = new_memory.do_not_ask[:8]
        return new_memory
