# Jack's Car Rental
# 10usd per rented car
# 2usd per car moved overnight
# Poisson random var returned and rented
# lambda for rental is 3 and 4
# lambda for return is 3 and 2
# no more than 20 cars at each location
# max 5 cars moved from one location to the other overnight
# gamma = 0.9
# continuing finite MDP
import numpy as np
import math
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

# one employee transfers the car for free
# from the first to the second location
VERSION = 1

MAP = {2: {}, 3: {}, 4: {}}


def prob(lam, n):
    global MAP
    if n not in MAP[lam]:
        MAP[lam][n] = lam**n / math.factorial(n) * math.exp(-lam)
    return MAP[lam][n]


class JCR:
    EARNED_BY_RENTAL = 10
    COST_OF_MOVING = -2
    PARKING_COST = -4
    GAMMA = 0.9
    MIN_REWARD = -100
    SHAPE = (21, 21)
    THETA = 0.0000001
    rows, cols = np.indices(SHAPE)
    STATES = list(zip(rows.flatten(), cols.flatten()))

    def __init__(self):
        self.__values = np.zeros(JCR.SHAPE)
        self.__policy = np.zeros(JCR.SHAPE, dtype=int)
        self.__rews0, self.__probs0 = self.__get_probs(3, 3)
        self.__rews1, self.__probs1 = self.__get_probs(4, 2)

    def __get_probs(self, rent_lambda, ret_lambda):
        rewards = np.zeros(JCR.SHAPE[0])
        probs = np.zeros(JCR.SHAPE)
        rent = 0
        rent_prob = 1
        while rent_prob > JCR.THETA:
            rent_prob = prob(rent_lambda, rent)
            for n in range(JCR.SHAPE[0]):
                actual_rent = min(n, rent)
                rewards[n] += JCR.EARNED_BY_RENTAL * rent_prob * actual_rent
            ret = 0
            ret_prob = 1
            while ret_prob > JCR.THETA:
                ret_prob = prob(ret_lambda, ret)
                for m in range(JCR.SHAPE[0]):
                    actual_rent = min(rent, m)
                    new_n = m + ret - actual_rent
                    new_n = max(new_n, 0)
                    new_n = min(JCR.SHAPE[0] - 1, new_n)
                    probs[m, new_n] += rent_prob * ret_prob
                ret += 1
            rent += 1
        return rewards, probs

    def __calculate_value(self, state, action):
        value = JCR.COST_OF_MOVING * abs(action)
        if VERSION == 2 and action > 0:
            value -= JCR.COST_OF_MOVING
        for to_state in JCR.STATES:
            reward = self.__rews0[state[0] - action] + \
                self.__rews1[state[1] + action]
            if to_state[0] > 10:
                reward += JCR.PARKING_COST
            if to_state[1] > 10:
                reward += JCR.PARKING_COST
            value += self.__probs0[state[0] - action, to_state[0]] * \
                self.__probs1[state[1] + action, to_state[1]] * \
                (reward + JCR.GAMMA * self.__values[to_state])
        return value

    def __policy_eval(self):
        while True:
            delta = 0
            for state in JCR.STATES:
                v = self.__values[state]
                action = self.__policy[state]
                self.__values[state] = self.__calculate_value(state, action)
                delta = max(delta, abs(v - self.__values[state]))
            if delta < JCR.THETA:
                break

    def __policy_improv(self):
        stable = True
        for state in JCR.STATES:
            old_action = self.__policy[state]
            low_bound = max(-5, -state[1])
            up_bound = min(6, state[0] + 1)
            actions = np.arange(start=low_bound, stop=up_bound, dtype=int)
            if len(actions) <= 1:
                continue
            vals = np.zeros(actions.shape)
            for i in range(len(actions)):
                action = actions[i]
                new_s0, new_s1 = state[0] - action, state[1] + action
                if new_s0 < 0 or new_s0 > 20 or new_s1 < 0 or new_s1 > 20:
                    vals[i] = JCR.MIN_REWARD
                else:
                    vals[i] = self.__calculate_value(state, action)
            self.__policy[state] = actions[np.argmax(vals)]
            if old_action != self.__policy[state]:
                stable = False
        return stable

    def calculate(self):
        i = 0
        policies = []
        while True:
            i += 1
            self.__policy_eval()
            if self.__policy_improv():
                break
            policies += [self.__policy.copy()]
        print(f'done! The optimal policy is:\n{self.__policy}')

        def __populate(axis, data, title):
            axis.imshow(data)
            axis.set_title(title)
            axis.set_xlabel('cars at first location')
            axis.set_ylabel('cars at second location')
            axis.xaxis.set_ticks(np.arange(JCR.SHAPE[0]))
            axis.yaxis.set_ticks(np.arange(JCR.SHAPE[1]))
            for i in range(JCR.SHAPE[0]):
                for j in range(JCR.SHAPE[1]):
                    axis.text(j, i, data[i, j], ha="center",
                            va="center", color="w")

        cols = 2
        rows = int(np.ceil((len(policies) + 1) / float(cols)))
        fig, axes = plt.subplots(nrows=rows, ncols=cols)
        for i in range(rows):
            for j in range(cols):
                if i * cols + j >= len(policies):
                    break
                __populate(axes[i][j], policies[i*cols+j],
                           f'policy {i*cols+j}')

        idx = len(policies)
        i = idx // cols
        j = idx % cols
        __populate(axes[i][j], self.__policy, 'optimal policy')

        fig = plt.figure()
        fig.suptitle('value function')
        ax = plt.axes(projection='3d')
        ax.xaxis.set_ticks(np.arange(JCR.SHAPE[0]))
        ax.set_xlabel('cars at first location')
        ax.yaxis.set_ticks(np.arange(JCR.SHAPE[1]))
        ax.set_ylabel('cars at second location')
        ax.plot_surface(JCR.rows, JCR.cols, self.__values)
        plt.show()


if __name__ == "__main__":
    jcr = JCR()
    jcr.calculate()
