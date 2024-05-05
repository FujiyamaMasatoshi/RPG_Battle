from enum import Enum, auto

# ゲーム状態の定義
class GameState(Enum):
    """ゲームと各ターン状態管理"""
    TURN_START = auto()      # ターン開始
    ACTION_ORDER = auto()  # コマンド選択
    POP_CHARACTER = auto()  # 行動キャラクタをpop
    TURN_NOW = auto()        # ターン中（各キャラ行動）
    TURN_END = auto()        # ターン終了
    GAME_END = auto()        # ゲーム終了
