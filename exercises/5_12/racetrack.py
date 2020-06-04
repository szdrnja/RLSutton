import numpy as np

class Generator:
    """
        Racetrack generator.

        Sourced with changes from
        https://towardsdatascience.com/solving-racetrack-in-reinforcement-learning-using-monte-carlo-control-bdee2aa4f04e
    """
    SIDE = 100
    INVALID = -1
    VALID = 0
    START = 1
    FINISH = 2

    def __init__(self):
        pass

    def __expand_hole_to_border(self, racetrack, start_cell, end_cell):
        """
            Expands the hole to a neighboring border
        """
        delta = 1
        while True:
            if start_cell[1] < delta or start_cell[0] < delta:
                start_cell = (0, 0)
                break

            if end_cell[1] + delta >= Generator.SIDE or end_cell[0] + delta >= Generator.SIDE:
                end_cell = (Generator.SIDE, Generator.SIDE)
                break

            delta += 1

        racetrack[start_cell[0]:end_cell[0], start_cell[1]:end_cell[1]] = Generator.INVALID

    def __mark_finish_states(self, racetrack):
        last_col = racetrack[:, -1]
        last_col[last_col == 0] = Generator.FINISH
        return racetrack

    def __mark_start_states(self, racetrack):
        last_row = racetrack[-1, :]
        last_row[last_row == 0] = Generator.START
        return racetrack

    def generate(self):
        '''
            The method call that actually generates the racetrack.

            :return:    matrix that represents the racetrack
        '''
        racetrack = np.zeros(
            shape=(Generator.SIDE, Generator.SIDE), dtype=int)
        MAX_HOLE_SIDE = 10

        valid_fraction = 1
        while valid_fraction > 0.5:
            center_x, center_y = np.random.randint(
                low=0 + MAX_HOLE_SIDE//2, high=Generator.SIDE-MAX_HOLE_SIDE, size=2)
            hole_width, hole_height = np.random.randint(
                low=0, high=MAX_HOLE_SIDE, size=2)
            start_x, start_y = max(
                0, center_x - hole_width//2), max(0, center_y - hole_height//2)
            end_x, end_y = min(Generator.SIDE, start_x +
                               hole_width), min(Generator.SIDE, start_y + hole_height)

            self.__expand_hole_to_border(racetrack, (start_x, start_y), (end_x, end_y))
            valid_fraction = len(racetrack[racetrack == 0]) / racetrack.size

        racetrack = self.__mark_start_states(racetrack)
        racetrack = self.__mark_finish_states(racetrack)

        return racetrack
