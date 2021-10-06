import numpy as np
import random
from scipy.misc import derivative

v_max = 10
p_c = 0.5
m = 10

y_max = 10
A = 10
p_w = 0.5

def phi(y):
    return int(A * np.sin( 2 * np.pi * y / y_max))

def u(t):
    return 1

def d(v):
    return np.random.choice(np.array([1, 0, -1]), p=np.array([(v/v_max) * (p_w / 2), 1 - (v/v_max) * (p_w), (v/v_max) * (p_w / 2)]).ravel())

def f(state, u, dt, t):
    y_new = state[0] + state[1] * dt
    has_crashed = random.uniform(0, 1) < ((np.abs(state[1]) - v_max) * p_c)/v_max
    if has_crashed:
        v_new = 0
    else:
        # d(cur v) or d(new v)
        v_new = state[1] + (1 / m) * (u - derivative(phi, state[0])) + d(state[1][0])

    return np.vstack((np.clip(round(y_new[0]), -y_max, y_max), np.clip(round(v_new[0]), -v_max, v_max)))

def h(state, input, t):
    return state[0] + np.random.normal(0, 0.5)

def r(state, input, next_state):
    # 

x = np.vstack((5, 10))











