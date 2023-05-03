# !/usr/bin/python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_file = "shrink_condition_1683091618.2209542.csv"
data_path = f"data/{data_file}"

df = pd.read_csv(data_path)
total_ps = np.array(df["shrink_condition"])


# fig, axs = plt.subplots(len(columns) - 1)
# fig.suptitle('Vertically stacked subplots')
# # Data to dictionary
# for i, robot_name in enumerate(columns[1:]):
#     robot_pairwise_data[robot_name] = {
#         'ts': np.array(df['ts']),
#         'data': np.array(df[robot_name]),
#     }

#     axs[i].plot(np.array(df[robot_name])[2000:])
plt.plot(total_ps)
plt.show()
# Plot data
