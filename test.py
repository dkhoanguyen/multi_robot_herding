import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2 * np.pi, 1000)
plt.axis([x[0], x[-1], -1, 1])      # disable autoscaling
for point in x:
    plt.plot(point, np.sin(2 * point), '.', color='b')
    plt.draw()
    plt.pause(0.01)
# plt.clf()                           # clear the current figure