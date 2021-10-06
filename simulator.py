from config import x, u, f, h

class Simulator:

    def __init__(self, dt, x, u, f, h, r, a, p, gamma) -> None:
        self.t = 0
        self.dt = dt # simulator timestep, e.g. 0.01
        self.x = x # initial state 
        self.u = u # u(t), input function
        self.f = f # f(x, u(t), dt, t), state update function
        self.h = h # h(x, u(t), t), observation function
        self.r = r # r(s, a, s')
        self.a = a # set (list) of all actions
        self.p = p # p(s, a, s'), prob of transitioning given action, cur state
        self.gamma = gamma # float

    def step(self):
        self.x = f(self.x, self.u(self.t), self.dt, self.t)
        self.t += self.dt

    def get_state(self):
        return self.x

    def get_observation(self):
        return h(self.x, self.u(self.t), self.t)

    def get_time(self):
        return self.t

    def value_iteration(self, termination_epsilon = 0.01):
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

                prev_value = value[state]
                policy[state], value[state] = max(action_values.items(), key=lambda x:x[1])
                value_delta = max(value_delta, abs(value[state] - prev_value))

    def policy_iteration(self, termination_epsilon = 0.01):
        policy, value, value_delta  = {}, {}, 0

        # Need to fix bug that occurs if two policies have equally good values and the algorithm switches between the two
        while True:
            # Policy Evaluation
            while value_delta > termination_epsilon:
                value_delta = 0
                for state in self.s:
                    state_value = 0
                    for next_state in self.s:
                        movement_prob = self.p(state, policy[state], next_state)
                        movement_reward = self.r(state, policy[state], next_state)
                        state_value += movement_prob * (movement_reward + self.gamma * value[next_state])

                    value_delta = max(value_delta, abs(value[state] - state_value))
                    value[state] = state_value

            # Policy Improvement
            is_policy_stable = True
            for state in self.s:
                from collections import defaultdict
                action_values = defaultdict(float)
                for action in self.a:
                    for next_state in self.s:
                        movement_prob = self.p(state, action, next_state)
                        movement_reward = self.r(state, action, next_state)
                        action_values[action] += movement_prob * (movement_reward + self.gamma * value[next_state])

                prev_state_policy = policy[state]
                policy[state], _ = max(action_values.items(), key=lambda x:x[1])
                if prev_state_policy != policy[state]:
                    is_policy_stable = False

            if is_policy_stable:
                break


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