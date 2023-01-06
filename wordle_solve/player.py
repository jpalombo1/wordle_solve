from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.stats import entropy  # type:ignore

from wordle_solve.constants import ALPH, ANSWER_LIST, WORD_SIZE, LetterState


@dataclass
class Player(ABC):
    """Player interactor to play Wordle.

    Attributes
    ----------
    random_seed (Optional[int]) : If set, randomness is fixes so same each run.
    """

    random_seed: Optional[int] = None

    def __post_init__(self) -> None:
        """Initialize other needed vars not in construction.

        Attributes
        ----------
        rsnp (np.random.RandomState) : Numpy random seed to fix randomness if needed.
        """
        self.rsnp = np.random.RandomState(seed=self.random_seed)
        self.reset_game()

    def reset_game(self) -> None:
        """Public method to reset player by resetting possible corpus.

        Attributes
        ----------
        corpus (list[str]) : List of all words left to use, initially all valid words.
        """
        self.corpus: list[str] = [word for word in ANSWER_LIST]

    @abstractmethod
    def guess_word(self, turn: int) -> str:
        """Guess word depending on player by human guess or comuter random/entropy.

        Parameters
        ----------
        turn (int) : Number of turns passed make turn.

        Returns
        -------
        (str) : Word guessed by human.
        """
        ...

    def reduce_guesses(self, word: str, states: list[LetterState]):
        """Reduce possible answers corpus by using guess word and proceeding states to eliminate.

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
        remove_idxs: list[int] = []
        match_letters: list[str] = [
            l for idx, l in enumerate(word) if states[idx] != LetterState.NONMATCH
        ]
        nonmatch_letters: list[str] = [
            l for idx, l in enumerate(word) if states[idx] == LetterState.NONMATCH
        ]
        matchletters: dict[str, int] = {
            ul: match_letters.count(ul) for ul in match_letters
        }
        nonmatchletters: dict[str, int] = {
            ul: nonmatch_letters.count(ul) for ul in nonmatch_letters
        }

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
                        state == LetterState.NONMATCH
                        and letter in possible_word
                        and word.count(letter) == 1
                    )
                    or (state == LetterState.WRONG_SPOT and letter == pos_letter)
                    or (state == LetterState.WRONG_SPOT and letter not in possible_word)
                    or (state == LetterState.RIGHT_SPOT and letter != pos_letter)
                ):
                    remove_idxs.append(word_idx)
                    break

        self.corpus = [
            answer for idx, answer in enumerate(self.corpus) if idx not in remove_idxs
        ]
        print(self.corpus)

    def recommend_guess(self, last_word: bool) -> None:
        """Method to print out recommended guess after entropy calculation.

        Parameters
        ----------
        last_word (bool) : If true, only get recommended guess from valid corpus since final guess, else can be any valid word to reduce corpus further.

        """
        rec_guess = self._get_entropy(last_word=last_word)
        print(f"Recommended Guess: {rec_guess}")

    def _get_entropy(self, last_word: bool = False) -> str:
        """Pick best word to divide corpus by maximum entropy. FOr case of size 2 or less corpus, use first entry to prevent repeat uses.

        First calculate number of times char appears at spot at corpus. e.g corpus 500 words a:[20,100,150,50,10] a appears 20 times at first spot, 100 at 2nd of word, etc.
        Then for each char of possible word, get # of greens by words with words at spot of corpus, yellows by # of times char appears in other spots for words in corpus, greys remaining corpus.
        Want to maximize entropy (or minimize differences in distribution green,yellow,grey) so pick word that maximizes entropy across chars amongst all possible words (not just corpus) to best split corpus.

        Parameters
        ----------
        last_word (bool) : If true, only get recommended guess from valid corpus since final guess, else can be any valid word to reduce corpus further.

        Returns
        -------
        str : Return best word for entropy.
        """
        if len(self.corpus) == 0:
            raise ValueError("No valid words left!")

        if len(self.corpus) in [1, 2]:
            return self.corpus[0]

        has_char_in_pos: dict[str, list[int]] = {}
        for char in ALPH:
            has_char_in_pos[char] = [
                sum(word[idx] == char for word in self.corpus)
                for idx in range(WORD_SIZE)
            ]

        diff_entropies: list[float] = []
        use_words = self.corpus if last_word else ANSWER_LIST
        for word in use_words:
            diff_entropy = 0.0
            for idx, char in enumerate(word):
                num_greens = has_char_in_pos[char][idx]
                num_yellows = sum(has_char_in_pos[char]) - num_greens
                num_greys = len(self.corpus) * WORD_SIZE - num_greens - num_yellows
                dist = np.array([num_greens, num_yellows, num_greys]).astype(float)
                dist /= np.sum(dist)
                diff_entropy += float(entropy(dist))
            diff_entropies.append(diff_entropy)
        best_entropy_idx = np.array(diff_entropies).argmax()
        best_word = use_words[best_entropy_idx]

        return best_word
