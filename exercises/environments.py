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
        _, r, terminal = super().step(action)
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


class KBandit(object):
    def __init__(self, k, mean=0, scale=1):
        self._q = np.random.normal(loc=mean, scale=scale, size=k)

    def reward(self, i):
        return self._q[i]

    def optimal(self):
        return np.argmax(self._q)


class KBanditNonStationary(KBandit):
    def __init__(self, k, mean=0, scale=1, update_scale=0.01):
        KBandit.__init__(self, k, mean, scale)
        self.__update_scale = update_scale

    def update(self):
        self._q += np.random.normal(scale=self.__update_scale, size=len(self._q))
