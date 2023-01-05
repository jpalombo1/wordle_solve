from dataclasses import dataclass

from wordle_solve.constants import WORD_SIZE, LetterState, check_word
from wordle_solve.player import Player

VALID_STATE_INPUT: list[int] = [1, 2, 3]


@dataclass
class Human(Player):
    """Player interactor to test reinforcement learning."""

    def guess_word(self, turn: int) -> str:
        """Call private method for human to enter guess.

        Parameters
        ----------
        turn (int) : Number of turns passed make turn.

        Returns
        -------
        (str) : Word guessed by human.
        """
        return self._check_word()

    def _check_state(self) -> list[LetterState]:
        """For assisted solver, give state of wordle game to help guessing.

        Make sure input states valid then convert to letter states.

        Returns
        -------
        list[LetterState] : List of letterStates derived from input values.
        """
        while True:
            try:
                response = input(
                    "Enter state ⬛->1,🟩->2,🟨->3 seperated by commas (e.g 1,1,2,1,3): "
                ).split(",")
                if len(response) != WORD_SIZE:
                    raise IndexError
                states = [int(state) for state in response]
                for state in states:
                    if state not in VALID_STATE_INPUT:
                        raise ValueError
                return [LetterState(state) for state in states]
            except IndexError:
                print(f"Need {WORD_SIZE} states only.")
            except ValueError:
                print(f"Invalid State values, Use only {VALID_STATE_INPUT}.")

    def _check_word(self) -> str:
        """Enter word and make sure word is a valid word.

        Returns
        -------
        str : Valid guess made by user.
        """
        while True:
            try:
                response = input(f"Please enter {WORD_SIZE} letter word: ")
                if len(response) != WORD_SIZE or not response.isalpha():
                    raise ValueError
                response = response.lower()
                valid_word = check_word(response)
                if not valid_word:
                    raise KeyError
                return response
            except ValueError:
                print(
                    f"Make sure your word containly ONLY {WORD_SIZE} alphabetical letters."
                )
            except KeyError:
                print("Please choose valid 5 letter word from dictionary!")
