# Joseph Palombo
from wordle_solve.board import Board
from wordle_solve.computer import Computer
from wordle_solve.human import Human

NUM_ITERS: int = 1


def main():
    """Main execution of agents and game."""
    set_word = "layer"
    comp_use_random = False
    turn_dist = {}
    board = Board(actual_word=set_word)
    p1 = Human()
    c1 = Computer(use_random=comp_use_random)
    for _ in range(NUM_ITERS):
        num_turns = board.play(c1)
        if num_turns not in turn_dist:
            turn_dist[num_turns] = 0
        turn_dist[num_turns] += 1
        print(turn_dist)
    board.play(p1)


if __name__ == "__main__":
    main()
