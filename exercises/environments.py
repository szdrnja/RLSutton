import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys, os

from utils import Color


class BaseEnv:
    def __init__(self, start, terminal, r, r_final):
        self._step = 0
        self._start_state = start
        self._terminal_state = terminal
        self._state = start
        self._r = r
        self._r_final = r_final

    def step(self, action):
        self._step += 1

    def restart(self):
        raise NotImplementedError

    def get_actions(self):
        raise NotImplementedError

    def get_action_shape(self):
        raise NotImplementedError

    def get_state_shape(self):
        raise NotImplementedError


class Gridworld(BaseEnv):
    START_VALUE = -1
    TERMINAL_VALUE = -2

    def __init__(self, map_shape, start, terminal, no_move=False, r=-1, r_final=0):
        super().__init__(start, terminal, r, r_final)
        if no_move:
            self._actions = np.arange(0, 9, dtype=int)
        else:
            self._actions = np.array([0, 1, 2, 3, 5, 6, 7, 8])
        self._map = np.zeros(map_shape, dtype=int)
        self._map[tuple(start)] = Gridworld.START_VALUE
        self._map[tuple(terminal)] = Gridworld.TERMINAL_VALUE

    def _get_d(self, action):
        dx = action // 3 - 1
        dy = action % 3 - 1
        return dx, dy

    def get_map(self):
        return self._map

    def restart(self):
        self._state = self._start_state
        return self._state

    def step(self, action):
        super().step(action)
        x0, y0 = self._state
        dx, dy = self._get_d(action)
        dimx, dimy = self._map.shape
        x1 = np.clip(x0 + dx, 0, dimx - 1)
        y1 = np.clip(y0 + dy, 0, dimy - 1)

        self._state = (x1, y1)
        terminal = self._state == self._terminal_state
        r = self._r if not terminal else self._r_final
        return self._state, r, terminal

    def get_actions(self):
        return self._actions

    def get_action_shape(self):
        return self._actions.shape

    def get_state_shape(self):
        return self._map.shape

    def get_color_scheme(self):
        return {
            0: Color.WHITE,
            Gridworld.START_VALUE: Color.YELLOW,
            Gridworld.TERMINAL_VALUE: Color.LIME,
            'cur': Color.BLUE}

    def visualize(self):
        plt.imshow(self._map)
        plt.show()


class WindyGridworld(Gridworld):
    def __init__(self, map_shape, wind, start, terminal, no_move=False, r=-1, r_final=0):
        super().__init__(map_shape, start, terminal, no_move, r, r_final)
        self._wind = np.zeros(tuple(np.concatenate((map_shape, [2]))))
        self._thrusts = []
        for state, thrust in wind:
            self._thrusts += [thrust]
            self._wind[state] = thrust
            self._map[state] = self._value_of_thrust(thrust)

    def _value_of_thrust(self, thrust):
        return 1000 * abs(thrust[0]) + abs(thrust[1])

    def step(self, action):
        dx, dy = self._wind[self._state]
        _, r, terminal = super().step(action)

        if dx or dy:
            x, y = self._state
            dimx, dimy = self._map.shape
            x = np.clip(x + dx, 0, dimx - 1)
            y = np.clip(y + dy, 0, dimy - 1)
            terminal = (x, y) == self._terminal_state
            r = self._r if not terminal else self._r_final
            self._state = (x, y)

        return self._state, r, terminal

    def get_color_scheme(self):
        scheme = super().get_color_scheme()
        for thrust in self._thrusts:
            scheme[self._value_of_thrust(thrust)] = (
                int(np.clip(thrust[0]*50, 0, 255)), 0,
                int(np.clip(thrust[1]*50, 0, 255)))

        return scheme


class StochasticWindyGridworld(WindyGridworld):
    def step(self, action):
        dx, dy = self._wind[self._state]
        _, r, terminal = super().step(action)

        if not terminal and (dx or dy):
            negate_dx = - np.sign(dx) * np.random.choice(np.arange(0, abs(dx)))
            negate_dy = - np.sign(dy) * np.random.choice(np.arange(0, abs(dy)))
            x = np.clip(x + negate_dx, 0, self._map.shape[0] - 1)
            y = np.clip(y + negate_dy, 0, self._map.shape[1] - 1)
            self._state = (x, y)

        terminal = self._state == self._terminal_state
        r = self._r if not terminal else self._r_final
        return self._state, r, terminal


class ObstacleGridworld(Gridworld):
    OBSTACLE_VALUE = -3

    def __init__(self, map_shape, obstacles, start, terminal, no_move=False, r=-1, r_final=0):
        super().__init__(map_shape, start, terminal, no_move, r, r_final)

        self._obstacles = obstacles
        for obstacle in obstacles:
            self._map[obstacle] = ObstacleGridworld.OBSTACLE_VALUE

    def step(self, action):
        previous_state = self._state
        _, r, terminal = super().step(action)

        if self._map[self._state] == ObstacleGridworld.OBSTACLE_VALUE:
            x0, y0 = previous_state
            x1, y1 = self._state
            if self._map[x0, y1] != ObstacleGridworld.OBSTACLE_VALUE:
                x1 = x0
            elif self._map[x1, y0] != ObstacleGridworld.OBSTACLE_VALUE:
                y1 = y0
            else:
                x1, y1 = x0, y0
            self._state = (x1, y1)
            terminal = self._state == self._terminal_state
            r = self._r if not terminal else self._r_final

        return self._state, r, terminal

    def get_color_scheme(self):
        scheme = super().get_color_scheme()
        scheme[ObstacleGridworld.OBSTACLE_VALUE] = Color.DARK_GRAY
        return scheme


class ChangingObstacleGridworld(ObstacleGridworld):
    def __init__(self, map_shape, obstacles, change_times,
                 start, terminal, no_move=False, r=-1, r_final=0):
        super().__init__(
            map_shape, obstacles[0], start, terminal, no_move, r, r_final)
        self._obstacles = obstacles
        self._obstacle_idx = 0
        self._change_times = change_times

    def change_obstacles(self):
        for ob in self._obstacles[self._obstacle_idx]:
            self._map[ob] = 0
        self._obstacle_idx += 1
        for ob in self._obstacles[self._obstacle_idx]:
            self._map[ob] = ObstacleGridworld.OBSTACLE_VALUE

    def step(self, action):
        _, r, terminal = super().step(state, action)
        if self._obstacle_idx < len(self._change_times) and \
                self._step > self._change_times[self._obstacle_idx]:
            self.change_obstacles()
        return self._state, r, terminal


class Canvas(BaseEnv):
    STATE_COLOR = 128
    def __init__(self, resolution, start, terminal, r=-1, r_final=0):
        super().__init__(start, terminal, r, r_final)
        self._canvas = np.zeros(resolution, dtype=np.uint8)
        # self._state_canvas = np.zeros((500, 500), np.int8)

    def get_canvas(self):
        return self._canvas

    def restart(self):
        self._state = self._start_state
        return self._state

    def get_color_scheme(self):
        return {0: Color.WHITE}

    def visualize(self):
        plt.imshow(self._canvas)
        plt.show()


class ObstacleCanvas(Canvas):
    OBSTACLE_COLOR = 255

    def __init__(self, resolution, start, terminal, obstacles_file, r=-1, r_final=0):
        super().__init__(resolution, start, terminal, r, r_final)
        obstacles = np.load(obstacles_file)
        cv2.fillPoly(self._canvas, obstacles, ObstacleCanvas.OBSTACLE_COLOR)

    def get_color_scheme(self):
        scheme = super().get_color_scheme()
        scheme[ObstacleCanvas.OBSTACLE_COLOR] = Color.GRAY
        return scheme


class RodManeuvering(ObstacleCanvas):
    OBSTACLES_FILE = os.path.join(os.path.dirname(sys.argv[0]), 'rod_maneuvering_obstacles.npy')
    WIDTH = 500
    HEIGHT = 500
    ROD_WIDTH = 10
    ROD_LENGTH = 170
    CELL_SIZE = 10
    START_STATE = np.array([71, 257, 87, 426])  # 169.7557
    TERMINAL_STATE = np.array([412, 242, 425, 72])  # 170.4963
    TERMINAL_STATE_N = 5629.69
    TERMINAL_STATE_K = -13.08
    STATE_COLOR = 50

    TRANSLATION_DIST = WIDTH / 20.
    HALF_TRANSLATION_DIST = TRANSLATION_DIST / 2.
    ANGLE = 10
    ROTATION_ANGLE = np.deg2rad(ANGLE)  # degrees
    HALF_ROTATION_SLOPE = np.tan(ANGLE / 2.)

    BACK_TRANSLATION = 0
    FORWARD_TRANSLATION = 1
    LEFT_TRANSLATION = 2
    RIGHT_TRANSLATION = 3
    LEFT_ROTATION = 4
    RIGHT_ROTATION = 5

    def __init__(self, r=-1, r_final=0):
        super().__init__((RodManeuvering.HEIGHT, RodManeuvering.WIDTH), RodManeuvering.START_STATE,
                         RodManeuvering.TERMINAL_STATE, RodManeuvering.OBSTACLES_FILE)
        self._actions = np.arange(6)
        self._bounding_rect = np.zeros_like(self._canvas)
        cv2.rectangle(self._bounding_rect, (0, 0), (RodManeuvering.HEIGHT-1, RodManeuvering.WIDTH-1), 1)
        self._terminal_state = np.zeros_like(self._canvas)
        cv2.line(self._terminal_state,
                tuple(RodManeuvering.TERMINAL_STATE[:2]),
                tuple(RodManeuvering.TERMINAL_STATE[2:]), 1)
        self._terminal_state_area = np.sum(self._terminal_state, dtype=float)
        self._drawing_board = np.zeros_like(self._canvas)

        # debug
        self._previous = None


    def _mid_point(self, p0, p1):
        return np.round((p0 + p1) / 2.)

    def _ext_state(self):
        a = np.array(self._state) / float(RodManeuvering.CELL_SIZE)
        a = np.round(a)
        return tuple(a)

    def _distance(self, p0, p1):
        return np.sqrt(pow(p1[0]-p0[0], 2) + pow(p1[1]-p0[1], 2))

    def _slope(self, p0, p1):
        dy = p1[1] - p0[1]
        dx = p1[0] - p0[0]
        if dx == 0:
            return np.inf #np.sign(dy) * np.inf
        return float(dy) / dx

    def _normal_slope(self, p0, p1):
        dy = p1[1] - p0[1]
        dx = p1[0] - p0[0]
        if dy == 0:
            return np.inf #np.sign(dx) * np.inf
        return - float(dx) / dy

    def _normalize(self, line):
        floor = np.array(np.floor(line), dtype=int)
        p0, p1 = floor[:2], floor[2:]
        x, y = p1
        possible_x, possible_y = np.arange(x, x+2, dtype=int), np.arange(y, y+2, dtype=int)
        possible_x = possible_x[(possible_x >= 0) & (possible_x < 500)]
        possible_y = possible_y[(possible_y >= 0) & (possible_y < 500)]
        possible_p1 = list(zip(
            np.repeat(possible_x, len(possible_y)),
            np.repeat(possible_y[None,...], len(possible_x), axis=0).flatten()))
        # check whether p1 intestects with obstacles?
        lns = np.array([self._distance(p0, p1) for p1 in possible_p1])
        dfs = np.abs(lns - RodManeuvering.ROD_LENGTH)
        p1 = possible_p1[np.argmin(dfs)]
        return np.concatenate((p0, p1))

    def _rotate(self, line, angle):
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle),  np.cos(angle)]])
        origin = self._mid_point(line[:2], line[2:])
        o = np.atleast_2d(origin)
        p0, p1 = np.atleast_2d(line[:2]), np.atleast_2d(line[2:])
        # @ - matrix multiplication
        p0 = np.squeeze((p0 - o) @ R + o)
        p1 = np.squeeze((p1 - o) @ R + o)
        return np.concatenate((p0, p1))

    def _ccw(self, p0, p1, p2):
        k0 = self._slope(p0, p1)
        k1 = self._slope(p1, p2)
        return k1 < k0

    def _get_possible_points(self, p, k, d):
        # Calculated a, b, c, using pythagoras and line formula
        x, y = p
        if k == 0:
            x0, x1 = x + d, x - d
            y0, y1 = y, y
        elif k == np.inf:
            x0, x1 = x, x
            y0, y1 = y + d, y - d
        else:
            dx = d / np.sqrt(1 + (k * k))
            dy = k * dx
            x0, x1 = x + dx, x - dx
            y0, y1 = y + dy, y - dy

        return (x0, y0), (x1, y1)

    def _draw_on_board(self, state):
        print(f'draw on board {state}')
        self._drawing_board[:] = 0
        cv2.line(self._drawing_board, tuple(state[:2]), tuple(state[2:]), RodManeuvering.STATE_COLOR, RodManeuvering.ROD_WIDTH)

    def _valid(self):
        # checks if state from _drawing_board is valid
        overlap = np.logical_and(self._drawing_board, self._canvas)
        if overlap.any():
            return False
        overlap = np.logical_and(self._drawing_board, self._bounding_rect)
        if overlap.any():
            return False
        return True

    def _terminal(self, state):
        # checks if state from _drawing_board is terminal
        def _almost_same(a, b, EPS=1e-5):
            return abs(a - b) < EPS
        k = self._slope(state[:2], state[2:])
        if not _almost_same(k, RodManeuvering.TERMINAL_STATE_K, RodManeuvering.HALF_ROTATION_SLOPE):
            return False
        n = state[1] - k * state[0]
        if not _almost_same(n, RodManeuvering.TERMINAL_STATE_N, RodManeuvering.HALF_TRANSLATION_DIST):
            return False
        overlap_area = np.logical_and(self._terminal_state, self._drawing_board).sum()
        return overlap_area / self._terminal_state_area > 0.8

    def restart(self):
        super().restart()
        self._previous = None

    def get_actions(self):
        return self._actions

    def get_action_shape(self):
        return self._actions.shape

    def get_state_shape(self):
        return (RodManeuvering.HEIGHT//RodManeuvering.CELL_SIZE,
            RodManeuvering.WIDTH//RodManeuvering.CELL_SIZE,
            RodManeuvering.HEIGHT//RodManeuvering.CELL_SIZE,
            RodManeuvering.WIDTH//RodManeuvering.CELL_SIZE)

    def step(self, action):
        super().step(action)
        if action not in self._actions:
            return self._state, self._r, False
        if action == RodManeuvering.BACK_TRANSLATION or action == RodManeuvering.FORWARD_TRANSLATION:
            p0, p1 = self._state[:2], self._state[2:]
            slope = self._slope(p0, p1)
            p00, p01 = self._get_possible_points(p0, slope, RodManeuvering.TRANSLATION_DIST)
            p10, p11 = self._get_possible_points(p1, slope, RodManeuvering.TRANSLATION_DIST)
            d0, d1 = self._distance(p1, p00), self._distance(p1, p01)
            if action == RodManeuvering.BACK_TRANSLATION and d0 > d1 or \
                action == RodManeuvering.FORWARD_TRANSLATION and d0 <= d1:
                next_state = np.concatenate((p00, p10))
            else: # action == RodManeuvering.FORWARD_TRANSLATION and d0 > d1
                # or action == RodManeuvering.BACK_TRANSLATION and d0 <= d1
                next_state = np.concatenate((p01, p11))
        elif action == RodManeuvering.LEFT_TRANSLATION or action == RodManeuvering.RIGHT_TRANSLATION:
            p0, p1 = self._state[:2], self._state[2:]
            slope = self._normal_slope(p0, p1)
            p00, p01 = self._get_possible_points(p0, slope, RodManeuvering.TRANSLATION_DIST)
            p10, p11 = self._get_possible_points(p1, slope, RodManeuvering.TRANSLATION_DIST)
            ccw0 = self._ccw(p0, p1, p10)
            if ccw0 and action == RodManeuvering.LEFT_TRANSLATION or not ccw0 and action == RodManeuvering.RIGHT_TRANSLATION:
                next_state = np.concatenate((p00, p10))
            else:
                # ccw0 and action == RodManeuvering.RIGHT_ROTATION or not ccw0 and action == RodManeuvering.LEFT_ROTATION
                next_state = np.concatenate((p01, p11))
        elif action == RodManeuvering.LEFT_ROTATION or action == RodManeuvering.RIGHT_ROTATION:
            angle = RodManeuvering.ROTATION_ANGLE
            if action == RodManeuvering.RIGHT_ROTATION:
                angle = -angle
            next_state = self._rotate(self._state, angle)

        next_state = self._normalize(next_state)
        r = self._r
        t = False
        self._draw_on_board(next_state)
        if self._valid():
            print(f'{self._state} -> {next_state}')
            self._state = next_state
            t = self._terminal(next_state)
            r = self._r_final

        return self._ext_state(), r, t

    def visualize(self):
        self._draw_on_board(self._state)
        overlap = self._canvas + self._drawing_board
        # if self._previous is not None:
        #     overlap += self._previous
        # self._previous = self._drawing_board.copy()
        plt.imshow(overlap)
        plt.show()




class KBandit(object):
    def __init__(self, k, mean=0, scale=1):
        self._q = np.random.normal(loc=mean, scale=scale, size=k)

    def reward(self, i):
        return self._q[i]

    def optimal(self):
        return np.argmax(self._q)


class KBanditNonStationary(KBandit):
    def __init__(self, k, mean=0, scale=1, update_scale=0.1):
        KBandit.__init__(self, k, mean, scale)
        self.__update_scale = update_scale

    def update(self):
        self._q += np.random.normal(scale=self.__update_scale)
