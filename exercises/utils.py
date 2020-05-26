import os
import pickle as pkl
import pygame
import numpy as np
import matplotlib.pyplot as plt


class Color:
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    LIME = (0, 255, 0)
    BLUE = (0, 0, 255)
    YELLOW = (255, 255, 0)
    CYAN = (0, 255, 255)
    MAGENTA = (255, 0, 255)
    SILVER = (192, 192, 192)
    GRAY = (128, 128, 128)
    MAROON = (128, 0, 0)
    OLIVE = (128, 128, 0)
    GREEN = (0, 128, 0)
    PURPLE = (128, 0, 128)
    TEAL = (0, 128, 128)
    NAVY = (0, 0, 128)
    DARK_GRAY = (64, 64, 64)
    # ALL = [
    #     BLACK, WHITE, RED, LIME,
    #     BLUE, YELLOW, CYAN, MAGENTA,
    #     SILVER, GRAY, MAROON, OLIVE,
    #     GREEN, PURPLE, TEAL, NAVY,
    # ]


class Visualizer:
    REFRESH_RATE = 10
    def __init__(self, m, color_scheme, caption='Visualizer',
                 line_color=(255, 255, 255), cell_size=5):
        self.__map = m
        self.__cell_size = cell_size
        self.__color_scheme = color_scheme
        self.__line_color = line_color
        self.__caption = caption
        self.__prev = None
        self.__active = False
        self._step = 0

    def __create_window(self):
        width = self.__map.shape[1] * self.__cell_size
        height = self.__map.shape[0] * self.__cell_size
        self.__display = pygame.display.set_mode((width, height))
        pygame.display.set_caption(self.__caption)
        for i in range(self.__map.shape[0]):
            for j in range(self.__map.shape[1]):
                self.__reset_cell((i, j))
        self.__active = True

    def __draw(self, cell, color):
        pygame.draw.rect(
            self.__display,
            color,
            ((cell[1] * self.__cell_size, cell[0] * self.__cell_size),
                (self.__cell_size, self.__cell_size)))
        pygame.draw.rect(
            self.__display,
            self.__line_color,
            ((cell[1] * self.__cell_size, cell[0] * self.__cell_size),
                (self.__cell_size, self.__cell_size)), 1)

    def __reset_cell(self, cell):
        cell_val = self.__map[cell]
        color = self.__color_scheme[cell_val]
        self.__draw(cell, color)

    def visualize(self, cell=None):
        if not self.__active:
            self.__create_window()
        if self._step % Visualizer.REFRESH_RATE == Visualizer.REFRESH_RATE - 1:
            for i in range(self.__map.shape[0]):
                for j in range(self.__map.shape[1]):
                    self.__reset_cell((i, j))
        self._step += 1
        if cell is not None:
            if self.__prev is not None:
                self.__reset_cell(self.__prev)
            if cell[0] < 0 or cell[0] >= self.__map.shape[0] \
                    or cell[1] < 0 or cell[1] >= self.__map.shape[1]:
                return
            color = self.__color_scheme['cur'] if 'cur' in self.__color_scheme \
                else Color.BLUE
            self.__draw(cell, color)
            self.__prev = cell
        pygame.display.update()

    # @staticmethod
    # def visualize_Q(Q):
    #     action_axis = self._action_idx % len(Q)
    #     val = np.max(Q, axis=action_axis)
    #     act = self._actions[np.argmax(Q, axis=action_axis)]
    #     _, axes = plt.subplots(ncols=2, nrows=1)
    #     axes[0].imshow(val)
    #     axes[1].imshow(act)
    #     plt.show(block=False)


class FileIO:
    @staticmethod
    def read_pkl(filepath):
        data = None
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as ifile:
                data = pkl.load(ifile)
        return data

    @staticmethod
    def dump_pkl(data, filepath):
        with open(filepath, 'wb') as ofile:
            pkl.dump(data, ofile)


class GreedyPolicy:
    def __init__(self, Q):
        self.__Q = Q

    def update_Q(self, Q):
        self.__Q = Q

    def __call__(self, state):
        return np.argmax(self.__Q[tuple(state)])


class EpsilonGreedyPolicy(GreedyPolicy):
    def __init__(self, Q, actions, e_checkpoints, s_checkpoints=[]):
        super().__init__(Q)
        slopes = []
        if len(e_checkpoints) != 1 and len(e_checkpoints) - 1 != len(s_checkpoints):
            raise Exception('s_checkpoints needs to have one' +
                            ' less element than e_checkpoints')
        s_checkpoints = [0] + s_checkpoints
        for i in range(len(e_checkpoints) - 1):
            e0, e1 = e_checkpoints[i:i+2]
            s0, s1 = s_checkpoints[i:i+2]
            slopes += [(e1-e0)/(s1-s0)]
        self.__e = e_checkpoints
        self.__s = s_checkpoints
        self.__slopes = slopes
        self.__actions = actions

    def epsilon(self, step):
        i = 0
        while i < len(self.__e) - 1 and self.__s[i + 1] < step:
            i += 1
        if i < len(self.__slopes):
            slope = self.__slopes[i]
            return self.__e[i] + slope * (step - self.__s[i])
        else:
            return self.__e[-1]

    def __call__(self, state, step, forced=False):
        e = self.epsilon(step)
        optimal_action = super().__call__(state)
        if np.random.random() > e or forced:
            action = optimal_action
        else:
            action = np.random.randint(low=0, high=len(self.__actions))

        if action == optimal_action:
            p = 1 - e + e / len(self.__actions)
        else:
            p = e / len(self.__actions)

        return action, p
