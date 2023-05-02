#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np

data = []
with open("data/gathered_flock.txt", "r") as f:
    all_data = f.readlines()
for data_point in all_data:
    data.append(float(data_point))

plt.plot(data)
plt.show( )
