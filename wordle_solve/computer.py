from wordle_solve.player import Player
from dataclasses import dataclass


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
