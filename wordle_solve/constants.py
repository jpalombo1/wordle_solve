from enum import Enum
from pathlib import Path

WORD_SIZE = 5
TOKENS = ["🔲", "⬛", "🟩", "🟨"]
WORD_PATH = Path(__file__).parent / "data" / "wordle_ist.csv"
ANSWER_PATH = Path(__file__).parent / "data" / "wordle_ist.csv"


class LetterState(Enum):
    """Enumeration of token states."""

    BLANK = 0
    NONMATCH = 1
    RIGHT_SPOT = 2
    WRONG_SPOT = 3
