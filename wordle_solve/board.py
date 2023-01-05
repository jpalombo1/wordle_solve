from dataclasses import dataclass
from typing import Optional

import numpy as np

from wordle_solve.constants import ALPH, ANSWER_LIST, TOKENS, WORD_SIZE, LetterState
from wordle_solve.human import Human
from wordle_solve.player import Player


@dataclass
class Board:
    """Board Object containing game.

    Attributes
    ----------
    max_turns (int) : Max attempts to guess word in game.
    random_seed (Optional[int]) : If set, randomness is fixes so same each run.
    actual_word (Optional[int]) : If set, an actual word is set, else a random valid answer is chosen.
    """

    max_turns: int = 6
    random_seed: Optional[int] = None
    actual_word: Optional[str] = None

    def __post_init__(self):
        """Initialize other needed vars outside construction.

        Attributes
        ----------
        rsnp (np.random.RandomState) : Numpy random seed to fix randomness if needed.
        """
        self.rsnp = np.random.RandomState(seed=self.random_seed)
        self.reset()

    def reset(self) -> None:
        """Reset board, including reseting state and guesses.

        Attributes
        ----------
        state (list[list[LetterState]]) : List of states for each state in word for each word in game. Size WORD_SIZE x max turns
        guess_word_list (list[str]) : List of guesses used by player.
        """
        self.state: list[list[LetterState]] = [
            [LetterState.BLANK for _ in range(WORD_SIZE)] for _ in range(self.max_turns)
        ]
        self.guess_word_list: list[str] = [""] * self.max_turns

    def _get_actual_word(self) -> str:
        """Return actual word if set, else choose a random word from all answers.

        Attributes
        ----------
        actual_word (str) : Actual word known to class.

        Returns
        -------
        (str) Actual word returned.
        """
        if self.actual_word is None:
            self.actual_word = self.rsnp.choice(list(ANSWER_LIST))
        return self.actual_word

    def _make_board(self) -> str:
        """Private method to construct board with squares, words, alphabet left.

        Returns
        -------
        (str) Representation of board in string form with rows for each guess, token boxes, and status.
        """
        left_alphabet = self._get_alphabet()
        start_board = "\n" + "-" * (WORD_SIZE + 1) + "\n"
        for turn in range(self.max_turns):
            val_list = [TOKENS[colval.value] for colval in self.state[turn]]
            start_board += " ".join(val_list)
            start_board += "\n" + "  ".join(self.guess_word_list[turn])
            start_board += "\n"
        start_board += f"Letters Not Used: {' '.join(sorted(list(left_alphabet)))}\n"
        return start_board

    def visualize_game(self) -> None:
        """Show visually pleasing current board."""
        print(self._make_board())

    def _get_alphabet(self) -> set[str]:
        """Private function to get remaining alphabet not guessed yet out of all words.

        Gets all letters used by concat all guesses to one string and gets set for all unique letters.
        Then Subtracts set of all letters from used letters to get remaining letters.

        Returns
        -------
        (set[str]) : Unique letters not used yet.
        """
        used_alph = set(list("".join(self.guess_word_list)))
        return set(list(ALPH)) - used_alph

    def _compare_guess(self, player_guess: str, actual_word: str) -> list[LetterState]:
        """Get state sequence from guess compared to actual word.

        First loop over word to see if any exact matches and count instances, assign match state.
        Loop again skippng over matches letters, if guess letter not in word, declare letter nonmatch.
        If guess letter in word, check if # of times in guess > # times in actual word, if so declare 2nd instance nonmatch, declare others wrong match.
        e.g. Guess word enter, actual word sober, first pss 2nd e given match state, since real e's (1) < guess e's (2) 1st e in enter given nonmatch.

        Parameters
        ----------
        player_guess (str) : Word player has guessed.
        actual_word (str) : Actual correct word.
        """
        unique_chars, counts = np.unique(list(actual_word), return_counts=True)
        char_count_real = {uchar: num for uchar, num in zip(unique_chars, counts)}
        char_count_guess: dict[str, int] = {}
        state_arr = [LetterState.BLANK] * WORD_SIZE
        for pos, (g_letter, act_letter) in enumerate(zip(player_guess, actual_word)):
            if g_letter not in char_count_guess:
                char_count_guess[g_letter] = 0
            if g_letter == act_letter:
                char_count_guess[g_letter] += 1
                state_arr[pos] = LetterState.RIGHT_SPOT

        for pos, (g_letter, act_letter) in enumerate(zip(player_guess, actual_word)):
            if state_arr[pos] == LetterState.RIGHT_SPOT:
                continue

            char_count_guess[g_letter] += 1
            if (
                g_letter in actual_word
                and char_count_guess[g_letter] <= char_count_real[g_letter]
            ):
                state_arr[pos] = LetterState.WRONG_SPOT
                continue

            state_arr[pos] = LetterState.NONMATCH
        return state_arr

    def game_over(self, player_guess: str) -> bool:
        """End game when actual word is guessed, reset board. Actual word already set in class.

        Parameters
        ----------
        player_guess (str) : Word player has guessed.

        Returns
        -------
        (bool) : Returns true if guess is equal to actual word.
        """
        if self.actual_word == player_guess:
            self.reset()
            return True
        return False

    def play(self, p1: Player) -> int:
        """Iterates a game for a player.

        Resets player and board, then player guesses word. Word evaluated by board to get state, board state and guess list updated, player corpus updated.
        Board visualized and checks for game over on each subsequent guess until max turns.

        Parameters
        ----------
        p1 (Player) : Player class that plays game.

        Returns
        -------
        (int) Number of turns for game.
        """
        p1.reset_game()
        self.reset()
        actual_word = self._get_actual_word()
        for turn in range(self.max_turns):
            word_guess = p1.guess_word(turn)
            self.state[turn] = self._compare_guess(word_guess, actual_word)
            self.guess_word_list[turn] = word_guess
            p1.reduce_guesses(word_guess, self.state[turn])

            self.visualize_game()
            last_word = True if turn >= self.max_turns - 1 else False
            p1.recommend_guess(last_word=last_word)

            if self.game_over(word_guess):
                print(f"You Won in {turn+1} turns!")
                return turn + 1
        print(f"You lost!, actual word was {actual_word}")
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
            last_word = True if turn >= self.max_turns - 1 else False
            p1.recommend_guess(last_word=last_word)

            if self.state[turn] == [2] * WORD_SIZE:
                print("Game Over!")
                return
