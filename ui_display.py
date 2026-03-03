import html
import os
import re
import glob
import time
import csv
from typing import Optional

import streamlit as st
import streamlit.components.v1 as components

# ページ設定
st.set_page_config(
    page_title="会話ロボット - リアルタイム表示",
    page_icon="🤖",
    layout="wide",
)

# 定数
IMAGE_PATH = "experiment_image.jpg"
LOGS_BASE = "logs"
REFRESH_INTERVAL = 2  # 秒


# -----------------------------
# カスタムCSS
# -----------------------------

CUSTOM_CSS = """
<style>
/* Streamlit デフォルトのパディングを縮小 */
.block-container {
    padding-top: 1rem !important;
    padding-bottom: 0rem !important;
    max-width: 100% !important;
}
header[data-testid="stHeader"] {
    height: 2rem !important;
}

/* タイトルバー */
.title-bar {
    background: linear-gradient(90deg, #1a1a2e 0%, #16213e 100%);
    color: #e0e0e0;
    padding: 0.4rem 1rem;
    border-radius: 8px;
    margin-bottom: 0.5rem;
    font-size: 1.1rem;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ステータスバー */
.status-bar {
    display: flex;
    gap: 1rem;
    align-items: center;
    background: #f0f2f6;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    margin-bottom: 0.5rem;
    flex-wrap: wrap;
}
.status-item {
    display: flex;
    align-items: center;
    gap: 0.3rem;
    font-size: 0.9rem;
    white-space: nowrap;
}
.status-label {
    color: #666;
    font-weight: 500;
}
.status-value {
    font-weight: 700;
    color: #1a1a2e;
}
.p-bar-container {
    flex: 1;
    min-width: 120px;
    max-width: 200px;
    height: 8px;
    background: #ddd;
    border-radius: 4px;
    overflow: hidden;
}
.p-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.3s ease;
}

/* 画像カラム */
.image-col img {
    border-radius: 8px;
}
</style>
"""


# -----------------------------
# データ取得関数
# -----------------------------

def find_latest_run() -> Optional[str]:
    """logs/run_* の中で最新のディレクトリを返す。"""
    pattern = os.path.join(LOGS_BASE, "run_*")
    dirs = sorted(glob.glob(pattern))
    return dirs[-1] if dirs else None


def find_file_in_dir(directory: str, prefix: str, ext: str) -> Optional[str]:
    """指定ディレクトリからプレフィックスと拡張子でファイルを探す。"""
    pattern = os.path.join(directory, f"{prefix}*{ext}")
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def read_analysis_csv(csv_path: str) -> dict:
    """
    analysis_*.csv の最終データ行から状態を読む。
    戻り値: {"p_want_talk": float, "phase": str, "turn": int}
    """
    result = {"p_want_talk": 0.5, "phase": "---", "turn": 0}
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            last_row = None
            for row in reader:
                last_row = row
        if last_row:
            result["p_want_talk"] = float(last_row.get("P_WantTalk", 0.5))
            result["phase"] = last_row.get("Phase", "---")
            result["turn"] = int(last_row.get("Turn", 0))
    except Exception:
        pass
    return result


def read_conversation_log(log_path: str) -> list[dict]:
    """
    log_*.txt をパースして会話メッセージのリストを返す。
    各要素: {"role": "ai"|"user", "content": str, "timestamp": str}
    """
    messages = []
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            raw = f.read()

        # パターン: [HH:MM:SS] AI: ... / [HH:MM:SS] User: ...
        parts = re.split(r'(\[\d{2}:\d{2}:\d{2}\] (?:AI|User):)', raw)
        current_role = None
        current_ts = None

        for part in parts:
            part = part.strip()
            if not part:
                continue

            m = re.match(r'\[(\d{2}:\d{2}:\d{2})\] (AI|User):', part)
            if m:
                current_ts = m.group(1)
                current_role = "ai" if m.group(2) == "AI" else "user"
            else:
                if current_role and part:
                    messages.append({
                        "role": current_role,
                        "content": part,
                        "timestamp": current_ts or "",
                    })
                    current_role = None
    except Exception:
        pass
    return messages


# -----------------------------
# UI 描画関数
# -----------------------------

# フェーズ名マッピング（英語→日本語）
PHASE_MAP = {
    "SETUP": "初期設定",
    "INTRO": "挨拶・アイスブレイク",
    "SURROUNDINGS": "周囲の話題",
    "BRIDGE": "連想（橋渡し）",
    "DEEP_DIVE": "深掘り",
    "ENDING": "エンディング",
}


def render_status_bar(p: float, phase: str, turn: int) -> None:
    """ステータスをコンパクトな1行バーで表示する。"""
    phase_ja = PHASE_MAP.get(phase, phase)

    if p >= 0.70:
        label = "高"
        color = "#4caf50"
        icon = "🟢"
    elif p >= 0.40:
        label = "中"
        color = "#ff9800"
        icon = "🟡"
    else:
        label = "低"
        color = "#f44336"
        icon = "🔴"

    bar_width = max(0, min(100, int(p * 100)))

    status_html = (
        f'<div class="status-bar">'
        f'<div class="status-item">'
        f'<span class="status-label">P(話したい)</span>'
        f'<span class="status-value">{icon} {p:.2f} ({label})</span>'
        f'<div class="p-bar-container">'
        f'<div class="p-bar-fill" style="width: {bar_width}%; background: {color};"></div>'
        f'</div></div>'
        f'<div class="status-item">'
        f'<span class="status-label">フェーズ</span>'
        f'<span class="status-value">{html.escape(phase_ja)}</span>'
        f'</div>'
        f'<div class="status-item">'
        f'<span class="status-label">ターン</span>'
        f'<span class="status-value">{turn}</span>'
        f'</div></div>'
    )
    st.markdown(status_html, unsafe_allow_html=True)


CHAT_CSS = """\
<style>
html, body {
    margin: 0; padding: 0; height: 100%;
    font-family: "Source Sans Pro", sans-serif;
    overflow: hidden;
}
.chat-container {
    height: 100%;
    overflow-y: auto;
    padding: 0.5rem;
    box-sizing: border-box;
    background: #fafafa;
}
.chat-bubble {
    margin-bottom: 0.6rem;
    display: flex;
    align-items: flex-start;
    gap: 0.5rem;
}
.chat-bubble-ai { justify-content: flex-start; }
.chat-bubble-user { justify-content: flex-end; }
.chat-avatar {
    font-size: 1.4rem;
    flex-shrink: 0;
    width: 2rem;
    text-align: center;
}
.chat-msg {
    max-width: 80%;
    padding: 0.5rem 0.8rem;
    border-radius: 12px;
    font-size: 0.9rem;
    line-height: 1.5;
    word-wrap: break-word;
}
.chat-msg-ai {
    background: #e3f2fd;
    color: #1a1a2e;
    border-bottom-left-radius: 4px;
}
.chat-msg-user {
    background: #e8f5e9;
    color: #1a1a2e;
    border-bottom-right-radius: 4px;
}
.chat-ts {
    font-size: 0.7rem;
    color: #999;
    margin-top: 0.2rem;
}
.empty-msg {
    color: #999;
    text-align: center;
    padding: 2rem;
}
</style>
"""


def render_chat_log(messages: list[dict]) -> None:
    """会話ログをチャットバブル風HTMLで描画する。components.html で確実にレンダリング。"""
    if not messages:
        bubbles_html = '<p class="empty-msg">まだ会話がありません。</p>'
    else:
        bubbles = []
        for msg in messages:
            role = msg["role"]
            content = html.escape(msg["content"])
            ts = html.escape(msg["timestamp"])

            if role == "ai":
                bubbles.append(
                    f'<div class="chat-bubble chat-bubble-ai">'
                    f'<div class="chat-avatar">🤖</div>'
                    f'<div class="chat-msg chat-msg-ai">{content}'
                    f'<div class="chat-ts">{ts}</div></div></div>'
                )
            else:
                bubbles.append(
                    f'<div class="chat-bubble chat-bubble-user">'
                    f'<div class="chat-msg chat-msg-user">{content}'
                    f'<div class="chat-ts">{ts}</div></div>'
                    f'<div class="chat-avatar">🧑</div></div>'
                )

        bubbles_html = "\n".join(bubbles)

    full_html = (
        CHAT_CSS
        + '<div class="chat-container" id="chatContainer">'
        + bubbles_html
        + '</div>'
        + '<script>'
        + 'var el = document.getElementById("chatContainer");'
        + 'if (el) { el.scrollTop = el.scrollHeight; }'
        + '</script>'
    )
    components.html(full_html, height=500, scrolling=False)


# -----------------------------
# メイン
# -----------------------------

UI_READY_FLAG = "ui_ready.flag"


def main() -> None:
    # UI起動フラグを作成（bayes_v3.py への通知）
    if not os.path.exists(UI_READY_FLAG):
        with open(UI_READY_FLAG, "w", encoding="utf-8") as f:
            f.write("ready")

    # カスタムCSS注入
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # タイトルバー（コンパクト）
    st.markdown(
        '<div class="title-bar">🤖 会話ロボット - リアルタイム表示パネル</div>',
        unsafe_allow_html=True,
    )

    # 自動更新フラグをセッションに保持
    if "auto_refresh" not in st.session_state:
        st.session_state.auto_refresh = True

    # ログディレクトリの取得（1回だけ）
    run_dir = find_latest_run()

    # --- 2カラムレイアウト: 左=画像, 右=ステータス＋会話ログ ---
    col_image, col_main = st.columns([2, 3])

    # 左カラム: 画像
    with col_image:
        if os.path.exists(IMAGE_PATH):
            st.image(IMAGE_PATH, use_container_width=True)
        else:
            st.warning(f"画像が見つかりません: `{IMAGE_PATH}`")

    # 右カラム: ステータス＋会話ログ
    with col_main:
        # ステータスバー
        if run_dir:
            csv_path = find_file_in_dir(run_dir, "analysis_", ".csv")
            if csv_path and os.path.exists(csv_path):
                state = read_analysis_csv(csv_path)
                render_status_bar(
                    p=state["p_want_talk"],
                    phase=state["phase"],
                    turn=state["turn"],
                )
            else:
                st.info("CSVログが見つかりません。エージェントの起動を待っています…")
        else:
            st.info("`bayes_v3.py` が未起動です。エージェントを起動してください。")

        # 会話ログ
        if run_dir:
            log_path = find_file_in_dir(run_dir, "log_", ".txt")
            if log_path and os.path.exists(log_path):
                messages = read_conversation_log(log_path)
                render_chat_log(messages)
            else:
                render_chat_log([])
        else:
            render_chat_log([])

    # --- フッター（自動更新コントロール） ---
    col_toggle, col_info = st.columns([1, 5])
    with col_toggle:
        if st.button("⏸️ 停止" if st.session_state.auto_refresh else "▶️ 再開"):
            st.session_state.auto_refresh = not st.session_state.auto_refresh

    with col_info:
        if st.session_state.auto_refresh:
            st.caption(f"⟳ 自動更新中（{REFRESH_INTERVAL}秒ごと）")
        else:
            st.caption("⏸️ 自動更新を停止中")

    # 自動更新
    if st.session_state.auto_refresh:
        time.sleep(REFRESH_INTERVAL)
        st.rerun()


if __name__ == "__main__":
    main()
