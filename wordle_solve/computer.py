from dataclasses import dataclass

from wordle_solve.player import Player


@dataclass
class Computer(Player):
    """Player interactor to test reinforcement learning.

    Attributes
    ----------
    use_random (bool) Determines if to just pick random word or use entropy procedure.
    """

    use_random: bool = False

    def guess_word(self, turn: int) -> str:
        """On turn 0 use best opener, rest use random guess or guess to maximize entropy.

        Parameters
        ----------
        turn (int) : Number of turns passed make turn.

        Returns
        -------
        (str) : Return word either by random, entropy, or best.
        """
        if turn == 0:
            return "soare"
        if self.use_random:
            return self.rsnp.choice(self.corpus)
        return self._get_entropy()
