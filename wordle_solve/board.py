from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from wordle_solve.human import Human
from wordle_solve.player import Player
from wordle_solve.constants import (
    TOKENS,
    WORD_SIZE,
    LetterState,
    WORD_PATH,
    ANSWER_PATH,
)


@dataclass
class Board:
    """Object containing game."""

    tokens: list[str] = field(default_factory=lambda: TOKENS)
    word_size: int = WORD_SIZE
    max_turns: int = 6
    random_seed: Optional[int] = None
    set_guess: Optional[str] = None

    def __post_init__(self):
        """Initialize other needed vars outside construction, like possible move indexes and state from board dimension."""

        self.rsnp = np.random.RandomState(seed=self.random_seed)
        self.answer_word_list: list[str] = (
            pd.read_csv(ANSWER_PATH).to_numpy().flatten().tolist()
        )
        self.valid_word_list: list[str] = (
            pd.read_csv(WORD_PATH).to_numpy().flatten().tolist() + self.answer_word_list
        )
        self.reset()

    def reset(self) -> None:
        """Reset board."""
        self.state: list[list[LetterState]] = [
            [LetterState.BLANK for _ in range(self.word_size)]
            for _ in range(self.max_turns)
        ]
        self.guess_word_list: list[str] = [""] * self.max_turns

    def _get_guess(self) -> str:
        """Set guess or choose random word from possible answers."""
        if self.set_guess:
            return self.set_guess
        return self.rsnp.choice(list(self.answer_word_list))

    def _make_board(self) -> str:
        """Private method to construct board with squares, words, alphabet left."""
        self._get_alphabet()
        start_board = "\n" + "-" * (self.word_size + 1) + "\n"
        for turn in range(self.max_turns):
            val_list = [self.tokens[colval] for colval in self.state[turn]]
            start_board += " ".join(val_list)
            start_board += "\n" + "  ".join(self.guess_word_list[turn])
            start_board += "\n"
        start_board += f"Letters Not Used: {' '.join(sorted(list(self.left_alph)))}\n"
        return start_board

    def visualize_game(self):
        """Show visually pleasing current board."""
        print(self._make_board())

    def check_word(self, word: str) -> bool:
        """Make sure word is in valid word list."""
        if word in set(self.valid_word_list):
            return True
        return False

    def _get_alphabet(self):
        """Private function to get remaining alphabe not guessed yet."""
        alph = "abcdefghijklmnopqrstuvwxyz"
        used_alph = set(list("".join(self.guess_word_list)))
        self.left_alph = set(list(alph)) - used_alph

    def _compare_guess(self, player_guess: str) -> list[int]:
        """Get state sequence from guess compared to actual word.

        First loop over word to see if any exact matches and count instances, assign match state.
        Loop again skippng over matches letters, if guess letter not in word, declare letter nonmatch.
        If guess letter in word, check if # of times in guess > # times in actual word, if so declare 2nd instance nonmatch, declare others wrong match.
        e.g. Guess word enter, actual word sober, first pss 2nd e given match state, since real e's (1) < guess e's (2) 1st e in enter given nonmatch.
        """
        unique_chars, counts = np.unique(list(self.guess_word), return_counts=True)
        char_count_real = {uchar: num for uchar, num in zip(unique_chars, counts)}
        char_count_guess = {}
        state_arr = [0] * self.word_size
        for pos, (g_letter, act_letter) in enumerate(
            zip(player_guess, self.guess_word)
        ):
            if g_letter not in char_count_guess:
                char_count_guess[g_letter] = 0
            if g_letter == act_letter:
                char_count_guess[g_letter] += 1
                state_arr[pos] = self.RIGHT_SPOT

        for pos, (g_letter, act_letter) in enumerate(
            zip(player_guess, self.guess_word)
        ):
            if state_arr[pos] == self.RIGHT_SPOT:
                continue

            char_count_guess[g_letter] += 1
            if (
                g_letter in self.guess_word
                and char_count_guess[g_letter] <= char_count_real[g_letter]
            ):
                state_arr[pos] = self.WRONG_SPOT
                continue

            state_arr[pos] = self.NONMATCH
        return state_arr

    def game_over(self, player_guess: str):
        """End game when word is gueesed."""
        if self.guess_word == player_guess:
            self.reset()
            return True
        return False

    def play(self, p1: Player) -> int:
        """Iterates a game for a player.

        Resets player and board, then player guesses word. Word evaluated by board to get state, board state and guess list updated, player corpus updated.
        Board visualized and checks for gaem over on each subsequent guess until max turns.
        """
        p1.reset_game()
        self.reset()
        self.guess_word = self._get_guess()
        for turn in range(self.max_turns):
            word_guess = p1.guess_word(turn)
            self.state[turn] = self._compare_guess(word_guess)
            self.guess_word_list[turn] = word_guess
            p1.reduce_guesses(word_guess, self.state[turn])

            self.visualize_game()
            p1._recommend_guess()

            if self.game_over(word_guess):
                print(f"You Won in {turn+1} turns!")
                return turn + 1
        print(f"You lost!, actual word was {self.guess_word}")
        return turn + 2

    def play_outside(self, p1: Human):
        """Iterates a game for human only.

        Manually enter state of outside wordle board each turn.
        """
        p1.reset_game()
        self.reset()
        for turn in range(self.max_turns):
            word_guess = p1.guess_word(turn)
            self.state[turn] = p1._check_state()
            self.guess_word_list[turn] = word_guess
            p1.reduce_guesses(word_guess, self.state[turn])
            self.visualize_game()
            p1._recommend_guess()

            if self.state[turn] == [2] * self.word_size:
                print("Game Over!")
                return
