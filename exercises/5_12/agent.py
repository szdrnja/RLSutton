import json
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import pickle as pkl
import os
import time
from env import Env

# Off-policy Monte Carlo Control
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class Car:
    MIN_STEP = 1000
    CONVERGENCE_EP = 50
    GAMMA = 1

    def __init__(self, env):
        self.__env = env
        self.__step = 0
        state_shape = self.__env.get_state_shape()
        action_shape = self.__env.get_action_shape()
        state_action_shape = np.concatenate((state_shape, action_shape))
        self.__num_actions = len(action_shape)
        self.__state_action = np.random.random(state_action_shape) * 400 - 500
        self.__policy_init = False
        self.__counts = np.zeros(state_action_shape)

    def load_model(self, model_name):
        filepath = os.path.join(SCRIPT_DIR, f'models/{model_name}.pkl')
        if not os.path.isfile(filepath):
            print(f'ERROR: unable to load model - {filepath} not found')
            return False
        with open(
                os.path.join(SCRIPT_DIR, f'models/{model_name}.pkl'), 'rb') as ifile:
            self.__state_action = pkl.load(ifile)
        self.__initialize_policy()
        return True

    def __initialize_policy(self):
        self.__target_policy = np.zeros(
            self.__env.get_state_shape(), dtype=int)
        valid_cells = self.__env.get_valid_cells()
        for x, y in valid_cells:
            for velx in range(5):
                for vely in range(5):
                    possible_actions = self.__env.get_possible_actions(
                        velx, vely)
                    impossible_actions = np.setdiff1d(
                        np.arange(9), possible_actions)
                    self.__state_action[x, y, velx,
                                        vely][impossible_actions] = -1e10
                    self.__target_policy[x, y, velx, vely] \
                        = possible_actions[
                            np.argmax(
                                self.__state_action[x, y, velx, vely][possible_actions])
                    ]
        self.__policy_init = True

    def __episode(self):
        def __epsilon(step):
            return 0.1

        def __target_policy(state):
            return self.__target_policy[tuple(state)]

        def __behavior_policy(state):
            """
                Returns     action
                            probability
            """
            def probability(state, action, possible_actions, e):
                if __target_policy(state) in possible_actions:
                    if action == __target_policy(state):
                        # either deliberately (1 - epsilon)
                        # or we weren't greedy and picked one (epsilon * 1 / len(possible_actions))
                        return 1 - e + e / len(possible_actions)
                    else:
                        # we weren't greedy and picked one (epsilon * 1 / len(possible_actions))
                        return e / len(possible_actions)
                else:  # we coulnd't even be greedy because the target policy's action is not possible
                    return 1 / len(possible_actions)

            possible_actions = self.__env.get_possible_actions(
                state[2], state[3])
            e = __epsilon(self.__step)
            if np.random.rand() > e and __target_policy(state) in possible_actions:
                action = __target_policy(state)
            else:
                action = np.random.choice(possible_actions)

            return action, probability(state, action, possible_actions, e)

        clean = True
        state = self.__env.start()

        # generate episode
        episode = {'S': [state], 'A': [], 'R': [None], 'p': []}
        done = False
        T = 0
        while not done:
            action, probability = __behavior_policy(state)
            state, reward, done = self.__env.step(state, action)
            episode['A'].append(action)
            episode['R'].append(reward)
            episode['p'].append(probability)
            episode['S'].append(state)
            self.__step += 1
            T += 1

        G = 0
        W = 1
        for t in range(T - 1, -1, -1):
            G = episode['R'][t+1] + Car.GAMMA * G
            # states never repeat so there is no need to make
            # the check for first visit MC
            sa_idx = tuple(np.concatenate(
                [episode['S'][t], [episode['A'][t]]]))

            # ajust count
            self.__counts[sa_idx] += W
            self.__state_action[sa_idx] += \
                W / self.__counts[sa_idx] * (G - self.__state_action[sa_idx])

            # update target policy
            best_action = np.argmax(
                self.__state_action[tuple(episode['S'][t])])
            self.__target_policy[tuple(episode['S'][t])] = best_action
            if best_action != episode['A'][t]:
                clean = False
                break
            W /= episode['p'][t]

        return clean, T

    def learn(self, episode_count):
        counter = 0
        sum_T = 0
        if not self.__policy_init:
            self.__initialize_policy()
        for i in range(episode_count):
            clean, T = self.__episode()
            sum_T += T
            if i % 100 == 0:
                print(f'{i}/{episode_count} - {sum_T/100.}')
                sum_T = 0
            counter = counter + 1 if clean else 0
            if self.__step > Car.MIN_STEP and counter > Car.CONVERGENCE_EP:
                print(f'converged after step {self.__step}')
                break

    def play(self, tries=10):
        if not self.__policy_init:
            print('ERROR: Policy not initialized on play')
            return None
        state = self.__env.start()

        done = False
        total_reward = 0
        reset_counter = 0
        t = 0
        while not done:
            self.__env.visualize(state)
            time.sleep(0.1)
            action = self.__target_policy[tuple(state)]
            state, reward, done = self.__env.step(state, action)
            if state[2] == state[3] == 0:
                if t == 0:
                    reset_counter += 1
                    state = self.__env.reset()
                    if reset_counter == tries:
                        print(f'giving up - was reset to start {tries} times')
                        done = True
                t = 0
            else:
                t += 1
            total_reward += reward

        return total_reward

    def save(self, model_name):
        fpath = os.path.join(SCRIPT_DIR, f'models/{model_name}.pkl')
        with open(fpath, 'wb') as ofile:
            pkl.dump(self.__state_action, ofile)
        print(f'model saved to {fpath}')
