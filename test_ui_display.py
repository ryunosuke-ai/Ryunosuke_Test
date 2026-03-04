"""
ui_display.py の純粋なデータ処理関数のユニットテスト
Streamlit の描画部分はモック化し、ファイルI/O・パースロジックを検証する。
"""
import os
import csv
import tempfile
import unittest
import sys
from unittest.mock import MagicMock

# Streamlit をモック化してインポート
sys.modules["streamlit"] = MagicMock()
sys.modules["streamlit.components"] = MagicMock()
sys.modules["streamlit.components.v1"] = MagicMock()

import ui_display
from ui_display import (
    find_latest_run,
    find_file_in_dir,
    read_analysis_csv,
    read_conversation_log,
)

LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")


# ================================================================
# find_latest_run
# ================================================================
class TestFindLatestRun(unittest.TestCase):

    def test_returns_none_when_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs_dir = os.path.join(tmp, "logs_empty")
            os.makedirs(logs_dir)
            orig = ui_display.LOGS_BASE
            ui_display.LOGS_BASE = logs_dir
            result = find_latest_run()
            ui_display.LOGS_BASE = orig
        self.assertIsNone(result)

    def test_returns_latest_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs_dir = os.path.join(tmp, "logs")
            for d in ["run_20260101_120000", "run_20260102_130000", "run_20260103_140000"]:
                os.makedirs(os.path.join(logs_dir, d))
            orig = ui_display.LOGS_BASE
            ui_display.LOGS_BASE = logs_dir
            result = find_latest_run()
            ui_display.LOGS_BASE = orig
        self.assertIn("run_20260103_140000", result)

    def test_single_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs_dir = os.path.join(tmp, "logs")
            os.makedirs(os.path.join(logs_dir, "run_20260101_000000"))
            orig = ui_display.LOGS_BASE
            ui_display.LOGS_BASE = logs_dir
            result = find_latest_run()
            ui_display.LOGS_BASE = orig
        self.assertIn("run_20260101_000000", result)


# ================================================================
# find_file_in_dir
# ================================================================
class TestFindFileInDir(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_finds_existing_csv(self):
        path = os.path.join(self.tmp, "analysis_20260101_120000.csv")
        open(path, "w").close()
        self.assertEqual(find_file_in_dir(self.tmp, "analysis_", ".csv"), path)

    def test_returns_none_when_not_found(self):
        self.assertIsNone(find_file_in_dir(self.tmp, "nonexistent_", ".csv"))

    def test_returns_latest_when_multiple(self):
        p1 = os.path.join(self.tmp, "log_20260101.txt")
        p2 = os.path.join(self.tmp, "log_20260102.txt")
        for p in [p1, p2]:
            open(p, "w").close()
        self.assertEqual(find_file_in_dir(self.tmp, "log_", ".txt"), p2)


# ================================================================
# read_analysis_csv
# ================================================================
class TestReadAnalysisCsv(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def _write_csv(self, name, rows):
        path = os.path.join(self.tmp, name)
        fields = ["Timestamp", "Turn", "Phase", "Speaker", "ActionType", "P_WantTalk", "Text"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        return path

    def test_reads_last_row(self):
        path = self._write_csv("a.csv", [
            {"Timestamp": "10:00:00", "Turn": "1", "Phase": "INTRO",       "Speaker": "AI",   "ActionType": "",       "P_WantTalk": "0.50", "Text": "こんにちは"},
            {"Timestamp": "10:00:05", "Turn": "2", "Phase": "SURROUNDINGS","Speaker": "User", "ActionType": "NORMAL", "P_WantTalk": "0.45", "Text": "まあまあ"},
        ])
        r = read_analysis_csv(path)
        self.assertAlmostEqual(r["p_want_talk"], 0.45)
        self.assertEqual(r["phase"], "SURROUNDINGS")
        self.assertEqual(r["turn"], 2)

    def test_empty_csv_defaults(self):
        path = os.path.join(self.tmp, "empty.csv")
        with open(path, "w", encoding="utf-8") as f:
            f.write("Timestamp,Turn,Phase,Speaker,ActionType,P_WantTalk,Text\n")
        r = read_analysis_csv(path)
        self.assertAlmostEqual(r["p_want_talk"], 0.5)
        self.assertEqual(r["phase"], "---")
        self.assertEqual(r["turn"], 0)

    def test_missing_file_graceful(self):
        r = read_analysis_csv("/nonexistent/path.csv")
        self.assertEqual(r["phase"], "---")

    def test_all_phases_readable(self):
        for phase in ["SETUP", "INTRO", "SURROUNDINGS", "BRIDGE", "DEEP_DIVE", "ENDING"]:
            path = self._write_csv(f"{phase}.csv", [
                {"Timestamp": "10:00:00", "Turn": "1", "Phase": phase, "Speaker": "AI", "ActionType": "", "P_WantTalk": "0.60", "Text": "test"},
            ])
            self.assertEqual(read_analysis_csv(path)["phase"], phase)


# ================================================================
# read_conversation_log
# ================================================================
class TestReadConversationLog(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def _write(self, content, name="log.txt"):
        path = os.path.join(self.tmp, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def test_parses_ai_and_user(self):
        path = self._write(
            "[10:00:00] AI: こんにちは。\n"
            "[10:00:05] User: はい。\n"
            "[10:00:10] AI: よかった。\n"
        )
        msgs = read_conversation_log(path)
        self.assertEqual(len(msgs), 3)
        self.assertEqual(msgs[0]["role"], "ai")
        self.assertEqual(msgs[1]["role"], "user")
        self.assertEqual(msgs[2]["role"], "ai")

    def test_timestamps_extracted(self):
        path = self._write("[13:45:22] AI: テスト。\n")
        self.assertEqual(read_conversation_log(path)[0]["timestamp"], "13:45:22")

    def test_empty_file(self):
        path = self._write("")
        self.assertEqual(read_conversation_log(path), [])

    def test_missing_file(self):
        self.assertEqual(read_conversation_log("/nonexistent/log.txt"), [])

    def test_content_integrity(self):
        ai_msg = "今日はどんなお気持ちですか？"
        user_msg = "まあまあです。"
        path = self._write(
            f"[09:00:00] AI: {ai_msg}\n"
            f"[09:00:10] User: {user_msg}\n"
        )
        msgs = read_conversation_log(path)
        self.assertEqual(msgs[0]["content"], ai_msg)
        self.assertEqual(msgs[1]["content"], user_msg)


# ================================================================
# 統合テスト（実際の logs/ を使用）
# ================================================================
class TestIntegration(unittest.TestCase):

    def test_can_find_existing_run(self):
        if not os.path.exists(LOGS_DIR):
            self.skipTest("logs/ ディレクトリが存在しません")
        orig = ui_display.LOGS_BASE
        ui_display.LOGS_BASE = LOGS_DIR
        run = find_latest_run()
        ui_display.LOGS_BASE = orig
        self.assertIsNotNone(run)

    def test_can_read_existing_csv(self):
        if not os.path.exists(LOGS_DIR):
            self.skipTest("logs/ ディレクトリが存在しません")
        orig = ui_display.LOGS_BASE
        ui_display.LOGS_BASE = LOGS_DIR
        run = find_latest_run()
        ui_display.LOGS_BASE = orig
        if not run:
            self.skipTest("ランディレクトリが見つかりません")
        csv_path = find_file_in_dir(run, "analysis_", ".csv")
        if not csv_path:
            self.skipTest("CSVファイルが見つかりません")
        r = read_analysis_csv(csv_path)
        self.assertGreaterEqual(r["p_want_talk"], 0.0)
        self.assertLessEqual(r["p_want_talk"], 1.0)

    def test_can_read_existing_log(self):
        if not os.path.exists(LOGS_DIR):
            self.skipTest("logs/ ディレクトリが存在しません")
        orig = ui_display.LOGS_BASE
        ui_display.LOGS_BASE = LOGS_DIR
        run = find_latest_run()
        ui_display.LOGS_BASE = orig
        if not run:
            self.skipTest("ランディレクトリが見つかりません")
        log_path = find_file_in_dir(run, "log_", ".txt")
        if not log_path:
            self.skipTest("ログファイルが見つかりません")
        msgs = read_conversation_log(log_path)
        self.assertIsInstance(msgs, list)
        for m in msgs:
            self.assertIn("role", m)
            self.assertIn("content", m)
            self.assertIn("timestamp", m)


if __name__ == "__main__":
    unittest.main(verbosity=2)
