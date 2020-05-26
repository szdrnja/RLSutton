from pathlib import Path
import os, sys
import numpy as np

script_path = Path(sys.argv[0])
sys.path.append(os.fspath(script_path.parent.parent))
from utils import Visualizer, Color, EpsilonGreedyPolicy
from predictors import DynaQ, DynaQPlus
from environments import ObstacleCanvas, RodManeuvering
from agents import Agent

env = RodManeuvering()

# BACK_TRANSLATION = 0
# FORWARD_TRANSLATION = 1
# LEFT_TRANSLATION = 2
# RIGHT_TRANSLATION = 3
# LEFT_ROTATION = 4
# RIGHT_ROTATION = 5

env.visualize()
env.step(0)
env.visualize()

# sa_shape = np.concatenate((env.get_state_shape(), env.get_action_shape()))
# Q = np.zeros(sa_shape)
# actions = env.get_actions()
# predictor = DynaQ(Q, actions, 5, EpsilonGreedyPolicy(Q, actions, [0.1]), alpha=0.5)
# agent = Agent(env, predictor)
# agent.learn(1000, visualize=True)
