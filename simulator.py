from config import x, u, f, h

class Simulator:

    def __init__(self, dt, x, u, f, h, r, a, p, gamma) -> None:
        self.t = 0
        self.dt = dt
        self.x = x
        self.u = u
        self.f = f # 
        self.h = h # 
        self.r = r # r(s, a, s')
        self.a = a # set of all actions
        self.p = p # P(s, a, s'), prob of transitioning given action, cur state
        self.gamma = gamma

    def step(self):
        self.x = f(self.x, self.u(self.t), self.dt, self.t)
        self.t += self.dt

    def get_state(self):
        return self.x

    def get_observation(self):
        return h(self.x, self.u(self.t), self.t)

    def get_time(self):
        return self.t

    def value_iteration(self):
        policy, value, value_delta, termination_epsilon = {}, {}, 0
        while value_delta > termination_epsilon:
            value_delta = 0
            for state in self.s:
                from collections import defaultdict
                action_values = defaultdict(float)
                for action in self.a:
                    for next_state in self.s:
                        movement_prob = self.p(state, action, next_state)
                        movement_reward = self.r(state, action, next_state)
                        action_values[action] += movement_prob * (movement_reward + self.gamma * value[next_state])

                policy[state], updated_value = max(action_values.items(), key=lambda x:x[1])
                value_delta = max(value_delta, abs(value[state] - updated_value))
                value[state] = updated_value


sim = Simulator(0.1, x, u, f, h)
position, velocity, measured_velocity, time = [], [], [], []
for i in range(100):
    position.append(sim.get_state()[0])
    velocity.append(sim.get_state()[1])
    measured_velocity.append(sim.get_observation())
    time.append(sim.get_time())
    print(sim.get_state())
    sim.step()

import matplotlib.pyplot as plt
plt.plot(time, position)
plt.plot(time, velocity)
plt.plot(time, measured_velocity)
plt.xlabel("Time")
plt.show()