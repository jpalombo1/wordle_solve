# Joseph Palombo
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import entropy  # type:ignore


@dataclass  # type: ignore
class Player(ABC):
    """Player interactor to play Wordle."""

    word_size: int = 5
    random_seed: Optional[int] = None

    def __post_init__(self):
        """Initialize other needed vars not in construction."""
        self.NONMATCH = 1
        self.RIGHT_SPOT = 2
        self.WRONG_SPOT = 3
        self.possible_answers: list[str] = (
            pd.read_csv("data/wordle_answers.csv").to_numpy().flatten().tolist()
        )
        self.corpus: list[str] = self.possible_answers
        self.rsnp = np.random.RandomState(seed=self.random_seed)

    @abstractmethod
    def guess_word(self, turn: int):
        """Guess word depending on player by human guess or comuter random/entropy."""
        return

    def reduce_guesses(self, word: str, states: list[int]):
        """Reduce possible answers by using guess word and proceeding states to eliminate.

        Cases of multiletter matches.
        Case of if multiple matches of guess, eliminate all guesses with less letters than number of matches.
        Case of same guess letter in match and nonmatch states, if possible word has more of letter than guess word matches, eliminate word since guess word would have more matches if it was valid.
        Case of guess letter in nonmatch but not part of matches, if letter appears in possible word, eliminate it since actual word should have not contain that letter
        Case of single letters in guess.
        If letter in guess nonmatch, remove all words with that letter AS LONG AS if guess word only has 1 of letter DUE TO multiletter exceptions above.
        If letter of possible answer is same as guess at that place, but declared wrong spot, eliminate word
        If letter declared in wrong spot but not declared in possible word, eliminate word.
        If letter of possible and answer differs from guess but guess letter in right spot, eliminate word
        """
        remove_idxs = []

        match_letters = [
            l for idx, l in enumerate(word) if states[idx] != self.NONMATCH
        ]
        nonmatch_letters = [
            l for idx, l in enumerate(word) if states[idx] == self.NONMATCH
        ]
        matchletters = {ul: match_letters.count(ul) for ul in match_letters}
        nonmatchletters = {ul: nonmatch_letters.count(ul) for ul in nonmatch_letters}

        for word_idx, possible_word in enumerate(self.corpus):
            for (ml, mlc) in matchletters.items():
                if possible_word.count(ml) < mlc:
                    remove_idxs.append(word_idx)
                    break
                if ml in nonmatch_letters and possible_word.count(ml) > mlc:
                    remove_idxs.append(word_idx)
                    break
            for nml in nonmatchletters.keys():
                if nml not in matchletters and possible_word.count(nml) > 0:
                    remove_idxs.append(word_idx)
                    break
            for place, (letter, state) in enumerate(zip(word, states)):
                pos_letter = possible_word[place]
                if (
                    (
                        state == self.NONMATCH
                        and letter in possible_word
                        and word.count(letter) == 1
                    )
                    or (state == self.WRONG_SPOT and letter == pos_letter)
                    or (state == self.WRONG_SPOT and letter not in possible_word)
                    or (state == self.RIGHT_SPOT and letter != pos_letter)
                ):
                    remove_idxs.append(word_idx)
                    break

        self.corpus = [
            answer for idx, answer in enumerate(self.corpus) if idx not in remove_idxs
        ]
        print(self.corpus)

    def _get_entropy(self):
        """Pick best word to divide corpus by maximum entropy. FOr case of size 2 or less corpus, use first entry to prevent repeat uses.

        First calculate number of times char appears at spot at corpus. e.g corpus 500 words a:[20,100,150,50,10] a appears 20 times at first spot, 100 at 2nd of word, etc.
        Then for each char of possible word, get # of greens by words with words at spot of corpus, yellows by # of times char appears in other spots for words in corpus, greys remaining corpus.
        Want to maximize entropy (or minimize differences in distribution green,yellow,grey) so pick word that maximizes entropy across chars amongst all possible words (not just corpus) to best split corpus.
        """
        if len(self.corpus) <= 2:
            return self.corpus[0]

        has_char_in_pos = {}
        for char in "abcdefghijklmnopqrstuvwxyz":
            has_char_in_pos[char] = [
                sum(word[idx] == char for word in self.corpus)
                for idx in range(self.word_size)
            ]
        diff_entropies = []
        for word in self.corpus:
            diff_entropy = 0
            for idx, char in enumerate(word):
                num_greens = has_char_in_pos[char][idx]
                num_yellows = sum(has_char_in_pos[char]) - num_greens
                num_greys = len(self.corpus) - num_greens - num_yellows
                dist = np.array([num_greens, num_yellows, num_greys]).astype(float)
                dist /= np.sum(dist)
                diff_entropy += entropy(dist)
            diff_entropies.append(diff_entropy)
            best_entropy_idx = np.array(diff_entropies).argmax()
        best_word = self.corpus[best_entropy_idx]

        return best_word

    def _recommend_guess(self):
        rec_guess = self._get_entropy()
        print(f"Recommended Guess: {rec_guess}")

    def reset_game(self):
        """Public method to reset player."""
        self.__post_init__()


@dataclass
class Human(Player):
    """Player interactor to test reinforcement learning."""

    def guess_word(self, turn: int) -> str:
        """Call private method for human to enter guess."""
        return self._check_word()

    def _check_state(self) -> list[int]:
        """For assisted solver, give state of wordle game to help guessing."""
        while True:
            try:
                response = input(
                    "Enter state ⬛->1,🟩->2,🟨->3 seperated by commas (e.g 1,1,2,1,3): "
                ).split(",")
                if len(response) != self.word_size:
                    raise IndexError
                states = [int(state) for state in response]
                for state in states:
                    if state < 1 or state > 3:
                        raise ValueError
                return states
            except IndexError:
                print(f"Need {self.word_size} states only.")
            except ValueError:
                print("Invalid State values, Use only (1, 2, or 3) .")

    def _check_word(self) -> str:
        """Enter word and make sure word is a valid word."""
        while True:
            try:
                response = input(f"Please enter {self.word_size} letter word: ")
                if len(response) != self.word_size or not response.isalpha():
                    raise ValueError
                response = response.lower()
                valid_word = Board().check_word(response)
                if not valid_word:
                    raise KeyError
                return response
            except ValueError:
                print(
                    f"Make sure your word containly ONLY {self.word_size} alphabetical letters."
                )
            except KeyError:
                print("Please choose valid 5 letter word from dictionary!")


@dataclass
class Computer(Player):
    """Player interactor to test reinforcement learning."""

    use_random: bool = False

    def guess_word(self, turn: int) -> str:
        """On turn 0 use best opener, rest use random guess or guess to maximize entropy."""
        if turn == 0:
            return "soare"
        if self.use_random:
            return self.rsnp.choice(self.corpus)
        return self._get_entropy()


@dataclass
class Board:
    """Object containing game."""

    tokens: list[str] = field(default_factory=lambda: ["🔲", "⬛", "🟩", "🟨"])
    word_size: int = 5
    max_turns: int = 6
    random_seed: Optional[int] = None
    set_guess: Optional[str] = None

    def _get_guess(self):
        """Set guess or choose random word from possible answers."""
        if self.set_guess:
            self.guess_word = self.set_guess
        else:
            self.guess_word = self.rsnp.choice(list(self.answer_word_list))

    def __post_init__(self):
        """Initialize other needed vars outside construction, like possible move indexes and state from board dimension."""
        self.BLANK = 0
        self.NONMATCH = 1
        self.RIGHT_SPOT = 2
        self.WRONG_SPOT = 3
        self.state = [
            [self.BLANK for _ in range(self.word_size)] for _ in range(self.max_turns)
        ]
        self.rsnp = np.random.RandomState(seed=self.random_seed)

        self.answer_word_list: list[str] = (
            pd.read_csv("data/wordle_answers.csv").to_numpy().flatten().tolist()
        )

        self.valid_word_list: list[str] = (
            pd.read_csv("data/wordle_list.csv").to_numpy().flatten().tolist()
            + self.answer_word_list
        )
        self.guess_word_list = [""] * self.max_turns

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

    def reset(self):
        """Reset board."""
        self.guess_word_list = [""] * self.max_turns
        self.state = [
            [self.BLANK for _ in range(self.word_size)] for _ in range(self.max_turns)
        ]

    def play(self, p1: Player) -> int:
        """Iterates a game for a player.

        Resets player and board, then player guesses word. Word evaluated by board to get state, board state and guess list updated, player corpus updated.
        Board visualized and checks for gaem over on each subsequent guess until max turns.
        """
        p1.reset_game()
        self.reset()
        self._get_guess()
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


def main():
    """Main execution of agents and game."""
    word_size = 5
    set_word = None
    comp_use_random = False
    turn_dist = {}
    board = Board(word_size=word_size, set_guess=set_word)
    p1 = Human(word_size=word_size)
    c1 = Computer(word_size=word_size, use_random=comp_use_random)
    # for _ in range(10):
    #     num_turns = board.play(c1)
    #     if num_turns not in turn_dist:
    #         turn_dist[num_turns] = 0
    #     turn_dist[num_turns] += 1
    #     print(turn_dist)
    board.play_outside(p1)


if __name__ == "__main__":
    main()
