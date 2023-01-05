# Joseph Palombo
from wordle_solve.board import Board
from wordle_solve.human import Human
from wordle_solve.computer import Computer


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
