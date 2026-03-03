import streamlit as st
import threading
import queue
import time
import os
from bayes_v3 import MultimodalAgent

# --- ページ設定とスタイル ---
st.set_page_config(
    page_title="実験用対話システム", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 画面全体の見栄えを良くするためのCSS
st.markdown("""
<style>
    /* 全体の背景と文字色（システムのテーマに依存しすぎないように微調整） */
    .stApp {
        background-color: #f4f6f9;
        color: #333333;
    }
    
    /* ヘッダーやテキストの色を明示的に指定 */
    h1, h2, h3, p, div {
        color: #333333;
    }

    /* 画像とチャットボックスのコンテナの装飾 */
    div[data-testid="column"] {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border: 1px solid #eaeaea;
    }
    
    /* チャット入力・表示枠周辺のUI改善 */
    .stChatMessage {
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 15px;
        background-color: #f8f9fa; /* うっすらグレーを敷いて可読性向上 */
    }
    
    /* ボタンのスタイル強化 */
    .stButton>button {
        background-color: #007bff;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 24px;
        width: 100%;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        color: white;
        box-shadow: 0 2px 8px rgba(0,123,255,0.3);
    }
</style>
""", unsafe_allow_html=True)

# --- 状態管理 ---
# スレッド間でやり取りするためのグローバルキュー（st.session_stateは別スレッドから触るとエラーになるため）
if "global_msg_queue" not in st.session_state:
    st.session_state.global_msg_queue = queue.Queue()
    
# エイリアスとしてグローバル変数にも持たせておく（バックグラウンドスレッド用）
# Streamlitのリロード時に消えない工夫が必要だが、簡易的にはグローバル変数を使うのが一番確実
import builtins
if not hasattr(builtins, "bg_msg_queue"):
    builtins.bg_msg_queue = queue.Queue()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "agent_thread" not in st.session_state:
    st.session_state.agent_thread = None
if "is_running" not in st.session_state:
    st.session_state.is_running = False

# メッセージをUIに追加する関数 (バックグラウンドスレッドから呼ばれる)
def log_message(sender, text, msg_type):
    # st.session_state ではなくグローバルのキューに入れる（スレッドセーフ）
    builtins.bg_msg_queue.put({"sender": sender, "text": text, "type": msg_type})

# エージェントの状態をメインスレッドに伝える
def set_running_state(state: bool):
    builtins.bg_msg_queue.put({"type": "internal_state", "is_running": state})

# エージェントを実行するスレッド関数
def run_agent():
    try:
        agent = MultimodalAgent(image_path="experiment_image.jpg")
        
        # エージェントのメソッドをオーバーライド（モンキーパッチ）してUIにログを流す
        original_speak = agent.speak
        original_listen = agent.listen
        
        def patched_speak(text: str):
            log_message("🤖 ロボット", text, "ai")
            original_speak(text)  # 実際の音声合成も実行
            
        def patched_listen():
            log_message("System", "🎤 マイク待機中...", "sys")
            text = original_listen() # 実際の音声認識を実行（UIはブロックされない）
            if text:
                log_message("🧑 あなた", text, "user")
            else:
                log_message("System", "(沈黙)", "sys")
            return text
            
        agent.speak = patched_speak
        agent.listen = patched_listen
        
        agent.run()
        log_message("System", "✅ 会話が終了しました。", "sys")

    except Exception as e:
        log_message("System", f"❌ エラーが発生しました: {e}", "sys")
    finally:
        set_running_state(False)


# --- UI レイアウト ---
st.title("実験用対話システム UI")
st.markdown("---")

col1, col2 = st.columns([5, 4], gap="large")

with col1:
    st.subheader("🖼️ 対象画像")
    img_path = "experiment_image.jpg"
    if os.path.exists(img_path):
        st.image(img_path, use_container_width=True) # 古いバージョンでも動く無難な書き方に戻すか、明示的にwidth指定を外す
    else:
        st.warning("⚠️ `experiment_image.jpg` が見つかりません。同じフォルダに画像を配置してください。")

with col2:
    st.subheader("💬 会話ログ")
    
    # チャット表示領域用コンテナ
    chat_container = st.container(height=500, border=False)

# --- 履歴の描画（通常表示） ---
with chat_container:
    for msg in st.session_state.chat_history:
        if msg["type"] == "sys":
            st.markdown(f'<div style="color: gray; font-size: 0.85rem; text-align: center;">{msg["text"]}</div>', unsafe_allow_html=True)
        elif msg["type"] == "ai":
            with st.chat_message("assistant", avatar="🤖"):
                st.write(msg["text"])
        elif msg["type"] == "user":
            with st.chat_message("user", avatar="🧑"):
                st.write(msg["text"])

# --- リアルタイムループ (フリッカー防止) ---
# st.rerun() で画面全体を再描画するとチカチカするため、
# 実行中は最後にこのループに留まり、キューから来たら動的にブロック追記するアプローチ
if st.session_state.is_running:
    while True:
        try:
            # タイムアウト付きでキューを待つ（0.1秒）
            msg = builtins.bg_msg_queue.get(timeout=0.1)
            
            if msg["type"] == "internal_state":
                st.session_state.is_running = msg["is_running"]
                if not msg["is_running"]:
                    break # ループを抜けて次回の操作を待つ
            else:
                st.session_state.chat_history.append(msg)
                # 追記描画
                with chat_container:
                    if msg["type"] == "sys":
                        st.markdown(f'<div style="color: gray; font-size: 0.85rem; text-align: center;">{msg["text"]}</div>', unsafe_allow_html=True)
                    elif msg["type"] == "ai":
                        with st.chat_message("assistant", avatar="🤖"):
                            st.write(msg["text"])
                    elif msg["type"] == "user":
                        with st.chat_message("user", avatar="🧑"):
                            st.write(msg["text"])

        except queue.Empty:
            # メッセージがない場合は少し待ってからブラウザに通底させるため、
            # Streamlitの動作を阻害しないよう短いsleepを入れる
            time.sleep(0.1)
            # ここで強制終了検知や再描画要求がある場合、適宜抜ける処理を入れる

    # 会話が終了したら一度だけ再描画して状態を確定させる
    st.rerun()

# --- ボタンの配置（一番下に独立して配置） ---
st.markdown("---")
cols = st.columns([1, 2, 1])
with cols[1]:
    if st.button("▶️ 会話を開始する", disabled=st.session_state.is_running):
        st.session_state.is_running = True
        st.session_state.chat_history = [] # 履歴リセット

        # キューをクリア
        while not builtins.bg_msg_queue.empty():
            builtins.bg_msg_queue.get()

        # バックグラウンドスレッドでエージェントを起動
        thread = threading.Thread(target=run_agent, daemon=True)
        st.session_state.agent_thread = thread
        thread.start()
        st.rerun()
