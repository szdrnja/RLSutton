import numpy as np
import time
import os
import sys


class BaseAgent:
    def __episode(self):
        raise NotImplementedError

    def learn(self, iterations):
        raise NotImplementedError

    def test(self, iterations):
        raise NotImplementedError

    def load(self, model):
        raise NotImplementedError

    def save(self, model):
        raise NotImplementedError


class Agent(BaseAgent):
    def __init__(self, env, predictor, vis=None):
        self.__env = env
        self.__predictor = predictor
        self.__actions = env.get_actions()
        self.__vis = vis

    def __episode(self, learn=False, visualize=False, sleep=0.5):
        policy = self.__predictor.behavior_policy if learn else self.__predictor.optimal_policy
        state = self.__env.restart()
        if visualize and self.__vis is not None:
            self.__vis.visualize(state)
            time.sleep(sleep) if sleep is not None else input()

        action = policy(state)
        terminal = False
        total_r = 0
        worst_err = 0

        while not terminal:
            next_state, reward, terminal = self.__env.step(
                state, self.__actions[action])
            total_r += reward
            if visualize and self.__vis is not None:
                # print(f'{state} - {self.__actions[action]} - {next_state}')
                self.__vis.visualize(next_state)
                time.sleep(sleep) if sleep is not None else input()

            if learn:
                err, action = self.__predictor.learn(
                    state, action, next_state, reward, terminal)
                if err is not None:
                    worst_err = max([worst_err, abs(err)])
            else:
                action = policy(next_state)
            state = next_state

        return total_r, worst_err

    def learn(self, iterations, early=False, visualize=False, stopper=0):
        rs = []
        total_r = 0
        total_err = 0
        for i in range(iterations):
            r, td_error = self.__episode(
                learn=True, visualize=visualize)
            total_err += td_error
            total_r += r
            rs += [r]
            if i and i % 100 == 0:
                print(f'{i}/{iterations}: {total_r/100.} {total_err/100.}')
                if early and (total_err < 1e-3 * 100 or abs(total_err/100.-stopper) < 1e-3):
                    print('converged')
                    break
                total_err, total_r = 0, 0
        return rs, i

    def test(self, iterations=1, visualize=True):
        s = 0
        for _ in range(iterations):
            r, _ = self.__episode(visualize=visualize, sleep=0.5)
            s += r
        return float(s) / iterations

    def load(self, model):
        self.__predictor.load(
            os.path.join(os.path.dirname(sys.argv[0]), f'models/{model}.pkl'))

    def save(self, model):
        self.__predictor.save(
            os.path.join(os.path.dirname(sys.argv[0]), f'models/{model}.pkl'))
