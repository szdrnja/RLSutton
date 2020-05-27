import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from predictors import KBanditGreedyInc, KBanditGreedyAlpha, KBanditUCBInc, \
    KBanditUCBAlpha, KBanditGradient
from environments import KBanditNonStationary

STEPS = 10000
EPSILON = 0.1
ALPHA = 0.1
UPDATE_PERIOD = 100
COLORS = ['g', 'b', 'r', 'c', 'm', 'y', 'k', 'w']

bandit = KBanditNonStationary(k=10, mean=0, scale=0, update_scale=0.01)
predictors = [
    KBanditGreedyInc(10, 0, 0, EPSILON),
    KBanditGreedyAlpha(10, 0, 0, EPSILON, ALPHA),
    KBanditGradient(10, 0, 0, ALPHA),
    KBanditUCBInc(10, 0, 0, 2),
    KBanditUCBAlpha(10, 0, 0, 2, ALPHA),
    KBanditGreedyInc(10, 0, 0, EPSILON),
]

plt.ion()
fig = plt.figure()
fig.suptitle('Performance of different algorithms against the optimal value')
ax = fig.add_subplot(111)
ax.set_xlim([0, STEPS])
ax.set_ylim([-3, 3])
ax.set_xlabel('steps')
ax.set_ylabel('reward')
bandit_plotline, = ax.plot([], [], COLORS[0] + '-', label='optimal')
plots = []
for i in range(len(predictors)):
    plots.append(
        ax.plot([], [], COLORS[i + 1] + '-', label=type(predictors[i]).__name__)[0])
ax.legend()
optimal_r_sum = 0

for i in range(STEPS):
    for predictor in predictors:
        pick = predictor.pick(i)
        reward = bandit.reward(pick)
        predictor.update(pick, reward)

    optimal_r_sum += bandit.reward(bandit.optimal())
    bandit.update()

    if i % UPDATE_PERIOD == 0:
        bandit_plotline.set_xdata(np.append(bandit_plotline.get_xdata(), i))
        bandit_plotline.set_ydata(
            np.append(bandit_plotline.get_ydata(), optimal_r_sum / (i+1)))
        for predictor, plot in list(zip(predictors, plots)):
            plot.set_xdata(np.append(plot.get_xdata(), i))
            plot.set_ydata(np.append(plot.get_ydata(), predictor.avg()))
        fig.canvas.draw()
        fig.canvas.flush_events()

plt.show(block=True)
