import json
import numpy as np
import pickle as pkl
import pygame
import matplotlib.pyplot as plt
from racetrack import Generator
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from utils import Visualizer, Color

class Racetrack:
    """
        Racetrack wrapper used by the environment.
    """

    def __init__(self, track_file):
        with open(track_file, 'rb') as in_file:
            self.__track = pkl.load(in_file)

        self.__on_track = None
        self.__start = None
        self.__finish = None
        self.__valid = None

    def __in_bounds(self, x, y):
        return x >= 0 and y >= 0 and x < self.__track.shape[0] and y < self.__track.shape[1]

    def get_start(self):
        if self.__start is None:
            x, y = np.where(self.__track == Generator.START)
            self.__start = list(zip(x, y))
        return self.__start[np.random.randint(0, len(self.__start))]

    def get_shape(self):
        return self.__track.shape

    def get_valid(self):
        if self.__valid is None:
            x, y = np.where(self.__track != Generator.INVALID)
            self.__valid = list(zip(x, y))
        return self.__valid

    def get_track(self):
        return self.__track

    def on_track(self, x, y):
        """
            Whether the given location is on the track.
            On the track is the starting position an any other position
            on the actual track.
        """
        return self.__in_bounds(x, y) and self.__track[x, y] in [Generator.START, Generator.VALID]

    def finish(self, x0, y0, dx, dy):
        """
            Check if finish line is crossed.
        """
        if self.__finish is None:
            x, y = np.where(self.__track == Generator.FINISH)
            self.__finish = list(zip(x, y))

        x1, y1 = x0 - dx, y0 + dy
        x, y = x0, y0
        rc = True
        while True:
            if (x, y) in self.__finish:
                break
            if x > x1:
                x -= 1
            if (x, y) in self.__finish:
                break
            if y < y1:
                y += 1
            if (x, y) in self.__finish:
                break
            if x == x1 and y == y1:
                rc = False
                break

        return rc, x, y


class Env:
    MIN_VELOCITY = 0
    MAX_VELOCITY = 5

    def __init__(self, track):
        self.__track = track
        self.__vis = Visualizer(track.get_track(),
            {Generator.VALID: Color.WHITE,
             Generator.INVALID: Color.BLACK,
             Generator.FINISH: Color.LIME,
             Generator.START: Color.GRAY},
             cell_size=10)

    def get_valid_cells(self):
        return self.__track.get_valid()

    def get_state_shape(self):
        x, y = self.__track.get_shape()
        velocity_range = Env.MAX_VELOCITY - Env.MIN_VELOCITY + 1
        return [x, y, velocity_range, velocity_range]

    def get_action_shape(self):
        return [9]

    def get_possible_actions(self, velx, vely):
        """
            Return a numpy array of possible actions given the state.
        """
        actions = []
        for ax in range(-1, 2):
            for ay in range(-1, 2):
                new_velx, new_vely = velx + ax, vely + ay
                if new_velx >= 0 and new_vely >= 0 \
                        and not (new_velx == 0 and new_vely == 0) \
                        and new_velx < Env.MAX_VELOCITY and new_vely < Env.MAX_VELOCITY:
                    actions += [self.__get_idx_from_tuple((ax, ay))]
        return np.array(actions)

    def get_action_space(self):
        return np.arange(10)
        # return self.__action_space.copy()

    def __get_tuple_from_idx(self, action_idx):
        x = action_idx // 3 - 1
        y = action_idx % 3 - 1
        return (x, y)

    def __get_idx_from_tuple(self, action_tuple):
        x, y = action_tuple
        return 3 * (x + 1) + (y + 1)

    def visualize(self, state):
        self.__vis.visualize(state[:2])

    # Defaults
    def step(self, state, action):
        """
            Returns
                new_state:  (np.array) new location and velocity of the car
                reward:     reward
                done:       whether finish line reached
        """
        action = self.__get_tuple_from_idx(action)
        x0, y0, velx, vely = state

        finished, x1, y1 = self.__track.finish(x0, y0, velx, vely)
        on_track = self.__track.on_track(x1, y1)

        if not finished and not on_track:
            return self.reset(), -1, finished

        velx = np.clip(velx + action[0], Env.MIN_VELOCITY, Env.MAX_VELOCITY)
        vely = np.clip(vely + action[1], Env.MIN_VELOCITY, Env.MAX_VELOCITY)
        return np.array([x1, y1, velx, vely]), -1, finished

    def start(self):
        """
            Returns new state.
        """
        x, y = self.__track.get_start()
        dx, dy = 0, 0
        return np.array([x, y, dx, dy])

    def reset(self):
        """
            Resets the environment and returns a state.
        """
        return self.start()
