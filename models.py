"""データ構造定義: Enum / dataclass"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, List


class ActionType(str, Enum):
    SILENCE = "SILENCE"
    NORMAL = "NORMAL"
    DISCLOSURE = "DISCLOSURE"


class Phase(str, Enum):
    SETUP = "初期設定"
    INTRO = "挨拶とアイスブレイク"
    SURROUNDINGS = "画像についての話題"
    BRIDGE = "連想（体験や好みへの接続）"
    DEEP_DIVE = "エピソードの深掘り"
    ENDING = "エンディング"


@dataclass
class PhaseConfig:
    name: Phase
    instruction: str
    require_image: bool = False
    max_turns: int = 3


@dataclass
class Observation:
    user_text: Optional[str]
    action_type: ActionType
    minimal_reply: bool = False
    memory_flag: bool = False
    self_disclosure_flag: bool = False
    engagement_hint: Optional[str] = None


@dataclass
class MemoryUpdate:
    summary: str
    do_not_ask: List[str]
    stop_intent: bool = False
