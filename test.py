import numpy as np
import matplotlib.pyplot as plt

magic = 1
theta = np.linspace(0., magic * np.pi, 1000)
a, b = 0, 10.
print(a+b*theta)
plt.polar(theta, a+b*theta)
plt.show()