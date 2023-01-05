from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import entropy  # type:ignore


@dataclass
class Player(ABC):
    """Player interactor to play Wordle."""

    word_size: int = 5
    random_seed: Optional[int] = None

    def __post_init__(self):
        """Initialize other needed vars not in construction."""
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
