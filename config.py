import numpy as np
import random
from scipy.misc import derivative

v_max = 10
p_c = 5
m = 10

def phi(y):
    return -2 * np.sin(y)

def u(t):
    return 0.5 * np.sqrt(t)

def f(state, u, dt, t):
    y_new = state[0] + state[1] * dt
    has_crashed = random.uniform(0, 1) < ((np.abs(state[1]) - v_max) * p_c)/v_max
    if has_crashed:
        v_new = 0
    else:
        v_new = state[1] + (1 / m) * (u - phi(state[0])) + + np.random.normal(0, 0.1)

    return np.vstack((y_new, v_new))

def h(state, input, t):
    return state[0] + np.random.normal(0, 0.5)

x = np.vstack((5, 10))











