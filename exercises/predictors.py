import sys
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import heapq
from utils import EpsilonGreedyPolicy, GreedyPolicy, FileIO

script_path = Path(sys.argv[0])
sys.path.append(os.fspath(script_path.parent.parent))


class BasePredictor:
    def __init__(self, actions, behavior_policy, alpha, gamma, action_idx=-1):
        self._alpha = alpha
        self._gamma = gamma
        self._step = 0
        self._actions = actions
        self._b = behavior_policy
        self._action_idx = action_idx

    def _td_error(self, state, action, next_state):
        raise NotImplementedError

    def learn(self, state, action, next_state, r, terminal):
        self._step += 1

    def behavior_policy(self, state):
        a, _ = self._b(state, self._step)
        return a

    def optimal_policy(self, state):
        a, _ = self._b(state, 0, forced=True)
        return a

    def load(self, filepath):
        raise NotImplementedError

    def save(self, filepath):
        raise NotImplementedError

    def _visualize_Q(self, Q, action_axis, block):
        val = np.round(np.max(Q, axis=action_axis))
        act = self._actions[np.argmax(Q, axis=action_axis)]
        _, axes = plt.subplots(ncols=2, nrows=1)
        axes[0].imshow(val)
        axes[1].imshow(act)
        plt.show(block=block)

    def visualize_Q(self, block=False):
        raise NotImplementedError


class Sarsa(BasePredictor):
    def __init__(self, Q, actions, behavior_policy, n=0, alpha=0.5):
        super().__init__(actions,
                         behavior_policy,
                         alpha=alpha, gamma=1)
        self._Q = Q
        self._n = n
        self._buffer = {
            'S': np.array([(0, 0)]*(n+1)),
            'A': np.array([0]*(n+1)),
            'r': np.array([0]*(n+1))}
        discounts = np.repeat(1, n+1)
        self._discounts = np.power(discounts, np.arange(n+1))

    def _populate_buffer(self, state, action, next_state, r, terminal):
        idx = (self._step - 1) % (self._n+1)
        self._buffer['S'][idx] = state
        self._buffer['A'][idx] = action
        self._buffer['r'][idx] = r
        next_action, _ = self._b(next_state, self._step)
        return next_action

    def _get_idx(self, t):
        return t % (self._n + 1)

    def _gain(self, t, next_SA=None):
        gain = 0
        for i in range(self._n + 1):
            if t + i == self._step:
                break
            idx = self._get_idx(t + i)  # r at index is actually t+1
            gain += self._discounts[i] * self._buffer['r'][idx]
        if t + self._n == self._step - 1 and next_SA:
            gain += self._discounts[-1] * self._Q[next_SA]

        return gain

    def _update(self, t, next_SA=None):
        target = self._gain(t, next_SA)
        idx = self._get_idx(t)
        s0, a0 = self._buffer['S'][idx], self._buffer['A'][idx]
        SA = tuple(np.concatenate((s0, [a0])))
        err = target - self._Q[SA]
        self._Q[SA] += self._alpha * (err)
        return err

    def learn(self, state, action, next_state, r, terminal):
        super().learn(state, action, next_state, r, terminal)
        err = []
        next_action = self._populate_buffer(state, action, next_state, r, terminal)

        if self._step - 1 > self._n:
            next_SA = tuple(np.concatenate((next_state, [next_action])))
            err += [self._update(self._step - 1 - self._n, next_SA)]

        if terminal:
            for i in range(1, self._n + 1):
                err += [self._update(self._step - 1 - self._n + i)]
            self._step = 0

        return np.average(err) if err else 0, next_action

    def load(self, filepath):
        self._Q = FileIO.read_pkl(filepath)

    def save(self, filepath):
        FileIO.dump_pkl(self._Q, filepath)

    def visualize_Q(self, block):
        self._visualize_Q(self._Q, 2, block)


class OffPolicySarsa(Sarsa):
    def __init__(self, Q, actions, behavior_policy, n=0, alpha=0.5):
        super().__init__(Q, actions, behavior_policy, n, alpha)
        self._buffer['pi'] = np.array([0.]*(n+1))
        self._buffer['b'] = np.array([0.]*(n+1))
        self._pi = GreedyPolicy(self._Q)

    def __rho(self, t):
        rho = 1
        for i in range(1, self._n):
            if t + i == self._step - 1:
                break
            idx = self._get_idx(t + i)
            rho *= self._buffer['pi'][idx] / self._buffer['b'][idx]
            if not rho:
                break
        return rho

    def _update(self, t, next_SA=None):
        gain = self._gain(t, next_SA)
        idx = self._get_idx(t)
        s0, a0 = self._buffer['S'][idx], self._buffer['A'][idx]
        SA = tuple(np.concatenate((s0, [a0])))
        err = gain - self._Q[SA]
        self._Q[SA] += self._alpha * self.__rho(t) * err
        return err

    def _populate_buffer(self, state, action, next_state, r, terminal):
        idx = (self._step - 1) % (self._n + 1)
        self._buffer['S'][idx] = state
        self._buffer['A'][idx] = action
        self._buffer['r'][idx] = r
        next_action, p = self._b(next_state, self._step)
        optimal_action = self._pi(next_state)
        next_idx = self._step % (self._n + 1)
        self._buffer['pi'][next_idx] = int(optimal_action == next_action)
        self._buffer['b'][next_idx] = p
        return next_action


class PerDecisionOffPolicySarsa(OffPolicySarsa):
    def _gain(self, t, next_SA=None):
        gain = 0
        rho = 1
        for i in range(self._n + 1):
            if t + i == self._step:
                break
            idx = self._get_idx(t + i)  # r at index is actually t+1
            rho_t = self._buffer['pi'][idx] / self._buffer['b'][idx]
            rho *= rho_t
            # if not rho:
            #     break
            gain += self._discounts[i] * self._buffer['r'][idx] * rho

            if rho_t < 1:
                state = self._buffer['S'][idx]
                optimal_action = self._pi(state)
                gain += self._discounts[i] * (1 - np.clip(rho_t, 0, 1)) * \
                    self._Q[tuple(np.concatenate((state, [optimal_action])))]

        if t + self._n == self._step - 1 and next_SA and rho:
            gain += self._discounts[-1] * self._Q[next_SA]

        # idx = self._get_idx(t)
        # state, action = self._buffer['S'][idx], self._buffer['A'][idx]
        # gain += (1 - rho_t) * \
        #     self._Q[tuple(np.concatenate((state, [action])))]

        return gain

    def _update(self, t, next_SA=None):
        return super(OffPolicySarsa, self)._update(t, next_SA)


class QLearning(BasePredictor):
    def __init__(self, Q, actions, behavior_policy, alpha=0.5):
        super().__init__(actions, behavior_policy, alpha=alpha, gamma=1)
        self._pi = GreedyPolicy(Q)
        self._Q = Q

    def learn(self, state, action, next_state, r, terminal=None):
        super().learn(state, action, next_state, r, terminal)
        optimal_next_action = self.optimal_policy(next_state)
        SA = tuple(np.concatenate((state, [action])))
        next_SA = tuple(np.concatenate((next_state, [optimal_next_action])))
        td_error = r + self._Q[next_SA] - self._Q[SA]
        self._Q[SA] += self._alpha * td_error
        next_action, _ = self._b(next_state, self._step)
        return td_error, next_action

    def load(self, filepath):
        self._Q = FileIO.read_pkl(filepath)
        self._pi.update_Q(self._Q)
        self._b.update_Q(self._Q)

    def save(self, filepath):
        FileIO.dump_pkl(self._Q, filepath)

    def visualize_Q(self, block):
        self._visualize_Q(self._Q, 2, block)


class ExpectedSarsa(BasePredictor):
    def __init__(self, Q, actions, alpha=1):
        super().__init__(actions,
                         EpsilonGreedyPolicy(Q, actions, [0.1]),
                         alpha=alpha, gamma=1)
        self._Q = Q

    def learn(self, state, action, next_state, r, terminal):
        target = 0
        optimal_next_action, _ = self.optimal_policy(next_state)
        e = self._b.epsilon(self._step)
        for next_action in self._actions:
            next_SA = tuple(np.concatenate((next_state, [next_action])))
            if next_action == optimal_next_action:
                prob = 1 - e + e / len(self._actions)
            else:
                prob = e / len(self._actions)
            target += prob * self._Q[next_SA]

        SA = tuple(np.concatenate((state, [action])))
        td_error = r + target - self._Q[SA]
        self._Q[SA] += self._alpha * td_error
        next_action, _ = self._b(next_state, self._step)
        return td_error, next_action

    def load(self, filepath):
        self._Q = FileIO.read_pkl(filepath)

    def save(self, filepath):
        FileIO.dump_pkl(self._Q, filepath)

    def visualize_Q(self, block):
        self._visualize_Q(self._Q, 2, block)


class DoubleQLearning(BasePredictor):
    def __init__(self, Q1, Q2, actions, alpha=0.5):
        self.__merged_Q = Q1 + Q2
        super().__init__(actions,
                         EpsilonGreedyPolicy(self.__merged_Q, actions, [0.1]), alpha=alpha, gamma=1
                         )
        self._Q1, self._Q2 = Q1, Q2
        self._Q1_greedy = GreedyPolicy(Q1)
        self._Q2_greedy = GreedyPolicy(Q2)

    def learn(self, state, action, next_state, r, terminal):
        td_error = None

        q1 = np.random.rand() > 0.5
        next_action = self._Q1_greedy(state) \
            if q1 else self._Q2_greedy(state)
        sa = tuple(np.concatenate((state, [action])))
        next_sa = tuple(np.concatenate((next_state, [next_action])))
        if q1:
            td_error = r + self._gamma * self._Q2[next_sa] - self._Q1[sa]
            self._Q1[sa] += self._alpha * td_error
        else:
            td_error = r + self._gamma * self._Q1[next_sa] - self._Q2[sa]
            self._Q2[sa] += self._alpha * td_error
        self.__merged_Q[sa] = self._Q1[sa] + self._Q2[sa]

        next_action, _ = self._b(next_state, self._step)
        return td_error, next_action

    def load(self, filepath):
        self._Q1 = FileIO.read_pkl(filepath)
        self._Q2 = np.copy(self._Q1)

    def save(self, filepath):
        FileIO.dump_pkl(self._Q1, filepath)

    def visualize_Q(self, block):
        self._visualize_Q(self._Q1, 2, block)
        self._visualize_Q(self._Q2, 2, block)


class DynaQ(QLearning):
    EXTENDER = 100
    def __init__(self, Q, actions, n, behavior_policy, alpha=0.5):
        super().__init__(Q, actions, behavior_policy, alpha)
        self._n = n
        self._sasr_shape = tuple(np.concatenate((Q.shape, [3])))
        self._model = np.zeros(self._sasr_shape, dtype=int)
        self._observed = self.__get_add_on()
        self._idx = 0

    def __get_add_on(self):
        return np.ones((DynaQ.EXTENDER, 3), dtype=int) * -1

    def __extend_observed(self):
        self._observed = np.concatenate((self._observed, self.__get_add_on()))

    def _plan(self):
        td_error = 0
        if self._idx <= self._n:
            batch = self._observed[:self._idx]
        else:
            indices = np.random.choice(self._idx, self._n, False)
            batch = self._observed[indices]
        for sample in batch:
            SR = self._model[tuple(sample)]
            td_err, _ = super().learn(
                (sample[0], sample[1]), sample[2],
                (SR[0], SR[1]), SR[2])
            td_error += td_err
        return td_error

    def _act(self, state, action, next_state, r):
        td_error, next_action = super().learn(state, action, next_state, r)
        SA = tuple(np.concatenate((state, [action])))
        self._model[SA] = np.concatenate((next_state, [r]))
        if SA not in self._observed:
            self._observed[self._idx] = SA
            self._idx += 1
            if self._idx == len(self._observed):
                self.__extend_observed()

        return td_error, next_action

    def learn(self, state, action, next_state, r, terminal=None):
        td_error, next_action = self._act(state, action, next_state, r)
        td_error += self._plan()
        return td_error, next_action


class DynaQPlus(DynaQ):
    def __init__(self, Q, actions, n, behavior_policy, kappa=0.01, alpha=0.5):
        super().__init__(Q, actions, n, behavior_policy, alpha)
        self._timers = np.zeros(Q.shape, dtype=int)
        self._kappa = kappa

    def _plan(self):
        td_error = 0
        if self._idx <= self._n:
            batch = self._observed[:self._idx]
        else:
            indices = np.random.choice(self._idx, self._n, False)
            batch = self._observed[indices]
        for sample in batch:
            SR = self._model[tuple(sample)]
            r = SR[2] + self._kappa * np.sqrt(self._step - self._timers[tuple(sample)])
            td_err, _ = super(DynaQ, self).learn(
                (sample[0], sample[1]), sample[2],
                (SR[0], SR[1]), r)
            self._timers[tuple(sample)] = self._step - 1
            td_error += td_err
        return td_error


class DynaQPrioritized(DynaQ):
    class Node:
        def __init__ (self, p, s, a):
            self._p = p
            self.s = s
            self.a = a

        def __lt__(self, other):
            return self._p < other._p

    def __init__(self, Q, actions, n, behavior_policy, theta=0.01, alpha=0.5):
        super().__init__(Q, actions, n, behavior_policy, alpha)
        self._theta = theta
        self._heap = []
        heapq.heapify(self._heap)


    def _plan(self):
        td_error = 0
        for _ in range(self._n):
            if not len(self._heap):
                break
            node = heapq.heappop(self._heap)
            SR = self._model[tuple(np.concatenate((node.state, node.action)))]
            td_err, _ = super().learn((node.s), node.a, (SR[0], SR[1]), SR[2])
            td_error += td_err

        return td_error

    def learn(self, state, action, next_state, r, terminal=None):
        td_error, next_action = self._act(state, action, next_state, r)

        p = abs(td_error)
        if p > self._theta:
            heapq.heappush(DynaQPrioritized.Node(p, state, action))

        td_error += self._plan()

        return td_error, next_action



class KBanditBase(object):
    def __init__(self, k, mean, scale):
        if scale < 0:
            raise ValueError('scale must be > 0')
        if k <= 0:
            raise ValueError('k must be > 0')
        self._q = np.random.normal(loc=mean, scale=scale, size=k)
        self._k = k
        self.__sum = 0
        self.__n = 0

    def update(self, r):
        self.__sum += r
        self.__n += 1

    def reset(self):
        self.__sum, self.__n = 0, 0

    def avg(self):
        return self.__sum / self.__n if self.__n else 0

# e greedy
class KBanditGreedy(KBanditBase):
    def __init__(self, k, mean, scale, e):
        KBanditBase.__init__(self, k , mean, scale)
        if e < 0 or e >= 1:
            raise ValueError('e must be >= 0 and < 1')
        self.__e = e

    def pick(self):
        return np.argmax(self._q) \
            if np.random.rand() > self.__e \
            else np.random.randint(low=0, high=self._k)

class KBanditGreedyInc(KBanditGreedy):
    def __init__(self, k, mean, scale, e):
        KBanditGreedy.__init__(self, k, mean, scale, e)
        self.__n = np.zeros(k)

    def update(self, a, r):
        KBanditGreedy.update(self, r)
        self.__n[a] += 1

        self._q[a] += 1 / self.__n[a] * (r - self._q[a]) \
            if self.__n[a] != 0 else r - self._q[a]
        self.__n[a] += 1


class KBanditGreedyAlpha(KBanditGreedy):
    def __init__(self, k, mean, scale, e, alpha):
        KBanditGreedy.__init__(self, k, mean, scale, e)
        if alpha < 0 or alpha >= 1:
            raise ValueError('alpha must be >= 0 and < 1')
        self.__alpha = alpha

    def update(self, a, r):
        KBanditGreedy.update(self, r)
        self._q[a] += self.__alpha * (r - self._q[a])


class KBanditGradient(KBanditBase):
    def __init__(self, k, mean, scale, alpha):
        KBanditBase.__init__(self, k, mean, scale)
        if alpha < 0 or alpha >= 1:
            raise ValueError('alpha must be >= 0 and < 1')
        self.__h = np.zeros(k)
        self.__avg_r = 10
        self.__n = 0
        self.__probs = np.ones(k) / k
        self.__alpha = alpha

    def pick(self):
        return np.argmax(self.__probs)

    def update(self, a, r):
        KBanditGreedy.update(self, r)
        self.__n += 1
        self.__avg_r += 1 / self.__n * (r - self.__avg_r)

        updater = - self.__probs
        updater[a] = 1 - self.__probs[a]
        updater *= self.__alpha * (r - self.__avg_r)
        self.__h += updater

        s = sum(np.exp(self.__h))
        self.__probs = np.exp(self.__h) / s


class KBanditUCB(KBanditBase):
    def __init__(self, k, mean, scale, c):
        KBanditBase.__init__(self, k, mean, scale)
        if c < 0:
            raise ValueError('c must be >= 0')
        self.__c = c
        self._n = np.zeros(k)

    def pick(self, t):
        if sum(self._n) < self._k:
            for i in range(self._k):
                if self._n[i] == 0:
                    return i
        q = self._q + self.__c * np.sqrt(np.log(t) / self._n)
        return np.argmax(q)

class KBanditUCBInc(KBanditUCB):
    def __init__(self, k, mean, scale, c):
        KBanditUCB.__init__(self, k, mean, scale, c)

    def update(self, a, r):
        KBanditGreedy.update(self, r)
        self._n[a] += 1
        self._q[a] += 1 / self._n[a] * (r - self._q[a]) \
            if self._n[a] != 0 else r - self._q[a]

class KBanditUCBAlpha(KBanditUCB):
    def __init__(self, k, mean, scale, c, alpha):
        KBanditUCB.__init__(self, k, mean, scale, c)
        if alpha < 0 or alpha >= 1:
            raise ValueError('alpha must be >= 0 and < 1')
        self.__alpha = alpha

    def update(self, a, r):
        KBanditGreedy.update(self, r)
        self._n[a] += 1
        self._q[a] += self.__alpha * (r - self._q[a])
