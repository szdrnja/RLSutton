# Racetrack
# - velocity is horizontal and vertical in the range [0, 4]
# - actions are [-1, 0, 1] to horizontal or vertical velocity
# - both velocities cannot be zero during the race
# - state is the location of the vehicle and the velocities
# - rewards are -1 until the car reaches the finish line
# - if the car hits the track boundary it is returned to start
#   but the episode continues
# - before updating the cars position at each timestep check whether
#   the car intersects the track boundary

import matplotlib.pyplot as plt
import pickle as pkl
import os
import sys

from agent import Car
from env import Racetrack, Env
from racetrack import Generator

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_NAME = 'example'
RACETRACK_FILE = os.path.join(SCRIPT_DIR, 'racetrack.pkl')
NUMBER_OF_EPISODES_TO_LEARN = 50000
TEST = True


# ### To generate a new racetrack and save it:
# gen = Generator()
# racetrack = gen.generate()
# plt.imshow(racetrack)
# plt.show()
# with open(RACETRACK_FILE, 'wb') as ofile: pkl.dump(racetrack, ofile)

if __name__ == "__main__":
    env = Env(Racetrack(RACETRACK_FILE))
    agent = Car(env)

    if TEST:
        if agent.load_model(MODEL_NAME):
            tr = agent.play()
            print(f'test episode reward: {tr}')
    else:
        agent.learn(NUMBER_OF_EPISODES_TO_LEARN)
        agent.save(MODEL_NAME)
