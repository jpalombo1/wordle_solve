from enum import Enum
from pathlib import Path

import pandas as pd

WORD_SIZE: int = 5
TOKENS: list[str] = ["🔲", "⬛", "🟩", "🟨"]
WORD_PATH: Path = Path(__file__).parent / "data" / "wordle_list.csv"
ANSWER_PATH: Path = Path(__file__).parent / "data" / "wordle_answers.csv"

ANSWER_LIST: list[str] = pd.read_csv(ANSWER_PATH).to_numpy().flatten().tolist()
VALID_LIST: list[str] = (
    pd.read_csv(WORD_PATH).to_numpy().flatten().tolist() + ANSWER_LIST
)
ALPH: str = "abcdefghijklmnopqrstuvwxyz"


class LetterState(Enum):
    """Enumeration of token states."""

    BLANK = 0
    NONMATCH = 1
    RIGHT_SPOT = 2
    WRONG_SPOT = 3


def check_word(word: str) -> bool:
    """Make sure word is in valid word list.

    Returns
    -------
    (bool) Returns if word in valid words or not.
    """
    return True if word in VALID_LIST else False
