# !/usr/bin/python3

import os
import numpy as np
import pickle
import matplotlib.pyplot as plt


def extract_states(states_data, keyname: str):
    extracted_state = []
    for state_data in states_data:
        state = {keyname: state_data[keyname]}
        state.update({"ts": state_data["ts"]})
        extracted_state.append(state)
    return extracted_state


directory = "data/"
files = []
for filename in os.listdir(directory):
    if os.path.isfile(os.path.join(directory, filename)):
        files.append(filename)

# Print the list of files
ts_list = np.zeros((len(files),))
for idx, file in enumerate(files):
    if ".pickle" not in file:
        continue
    file = file.replace(".pickle", "")
    ts = float(file.split("_")[-1])
    ts_list[idx] = ts
max_idx = np.where(ts_list == ts_list.max())[0][0]
file = files[max_idx]

# Unpickle the list
with open(directory + file, 'rb') as f:
    raw_data = pickle.load(f)
config = raw_data["configuration"]
states_data = raw_data["data"]

robot_states = extract_states(states_data, "robot")
num_robots = robot_states[0]["robot"].shape[0]
target_indx = 5
d_one_and_other = np.empty((0, num_robots - 1))

# First row being the index
header = []
for idx in range(num_robots):
    if idx == target_indx:
        continue
    header.append(idx)
d_one_and_other = np.vstack((d_one_and_other, np.array(header)))

# Extract distance between robot 1 and other robots
for indx, robot_state in enumerate(robot_states):
    states = robot_state["robot"]
    all_d = []
    for robot_idx in header:
        d = np.linalg.norm(states[target_indx, :2] - states[robot_idx, :2])
        all_d.append(d)
    d_one_and_other = np.vstack((d_one_and_other, np.array(all_d)))

# Plot distance between robot 1 and 2
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(d_one_and_other[1:,6])
ax.axhline(y=60, color="black", linestyle="--", linewidth=1)
ax.axhline(y=300, color="black", linestyle="--", linewidth=1)
ax.set_xlabel("Timestep")
ax.set_ylabel("Distance")

plt.show()


