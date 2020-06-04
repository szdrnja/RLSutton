# Gambler's problem
# 100usd win
import numpy as np
import matplotlib.pyplot as plt

class GP:
    pH = 0.45
    pT = 1 - pH
    THETA = 1e-50
    GAMMA = 1
    GOAL = 100#usd

    def __init__(self):
        self.__values = np.zeros(GP.GOAL+1)
        self.__values[100] = 1
        self.__policy = np.zeros(GP.GOAL+1)

    def __populate_policy(self):
        for state in range(1, GP.GOAL):
            max_val, max_act = 0, 0
            for action in range(1, min(state, 100-state) + 1):
                val = self.__calculate_value(state, action)
                if val > max_val:
                    max_val, max_act = val, action
            self.__policy[state] = max_act

    def __calculate_value(self, state, action, print=False):
        loss = state - action
        win = state + action
        # reward of 1 is hidden in the state
        val = GP.pH * self.__values[win]
        val += GP.pT * self.__values[loss]
        return val

    def __value_iteration(self):
        iterations = 0
        sweep1, sweep2, sweep3, sweep32 = None, None, None, None

        while True:
            iterations += 1
            delta = 0
            for state in range(1, GP.GOAL):
                max_val = 0
                for action in range(1, min(state, GP.GOAL-state) + 1):
                    val = self.__calculate_value(state, action)
                    max_val = max(val, max_val)
                v = self.__values[state]
                self.__values[state] = np.round(max_val, 10)
                delta = max(delta, abs(v - self.__values[state]))

            if iterations == 1:
                sweep1 = self.__values.copy()
            elif iterations == 2:
                sweep2 = self.__values.copy()
            elif iterations == 3:
                sweep3 = self.__values.copy()
            elif iterations == 32:
                sweep32 = self.__values.copy()

            if delta < GP.THETA:
                break

        self.__sweeps = [sweep1, sweep2, sweep3, sweep32]
        print(f'iterations {iterations}')

    def run(self):
        COLORS = ['b', 'g', 'r', 'y', 'c', 'm', 'k', 'w']
        self.__value_iteration()
        self.__populate_policy()
        fig = plt.figure()
        ax = plt.axes()

        # i = 1
        # for sweep in self.__sweeps:
        #     if sweep is not None:
        #         ax.plot(np.arange(GP.GOAL+1), sweep, f'{COLORS[i]}-', label=f'sweep{i if i != 4 else 32}')
        #     i += 1
        ax.plot(np.arange(GP.GOAL+1), self.__values, f'{COLORS[0]}:', label='optimal')
        ax.set_xlabel('capital')
        ax.set_ylabel('value estimates')
        ax.legend()
        fig = plt.figure()
        fig.suptitle('Policy')
        ax = plt.axes()
        ax.plot(np.arange(GP.GOAL+1), self.__policy)
        ax.set_xlabel('capital')
        ax.set_ylabel('stake')
        plt.show()

if __name__ == "__main__":
    gp = GP()
    gp.run()
