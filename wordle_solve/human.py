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
