from config import x, u, f, h

class Simulator:

    def __init__(self, dt, x, u, f, h) -> None:
        self.t = 0
        self.dt = dt
        self.x = x
        self.u = u
        self.f = f
        self.h = h

    def step(self):
        self.x = f(self.x, self.u(self.t), self.dt, self.t)
        self.t += self.dt

    def get_state(self):
        return self.x

    def get_observation(self):
        return h(self.x, self.u(self.t), self.t)

    def get_time(self):
        return self.t


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