# !/usr/bin/python3

import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from matplotlib.patches import Patch
from matplotlib.lines import Line2D


class DataAnalyser(object):
    def __init__(self, data_name=None,
                 load_latest=True):
        self._raw_data = {}
        self._config = {}
        self._states_data = {}

        all_data = []
        all_mean = []
        for data in range(9,14):
            directory = f"data/30_animals/{data}/"
            if load_latest or data_name is None or data_name == "":
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

                self._all_herd_states = {}
                self._all_shepherd_states = {}

                for idx, file in enumerate(files):
                    if ".pickle" not in file:
                        continue
                    ts = ts_list[idx]
                    with open(directory + file, 'rb') as f:
                        raw_data = pickle.load(f)
                        config = raw_data['configuration']
                        states_data = raw_data['data']

                        herd_states = self.extract_states("herd", states_data)
                        shepherd_states = self.extract_states("shepherd" , states_data)
                        
                        self._all_herd_states[ts] = herd_states
                        self._all_shepherd_states[ts] = shepherd_states

            all_isolated_rate = []
            for ts, _ in self._all_herd_states.items():
                herd_states = self._all_herd_states[ts]
                shepherd_states = self._all_shepherd_states[ts]
                isolation_rate, first_zero = self.extract_isolation_rate(herd_states,shepherd_states)
                if first_zero != 0:
                    all_isolated_rate.append(first_zero/30)
            median = np.median(all_isolated_rate)
            all_mean.append(median)
            all_data.append(all_isolated_rate)

        plt.boxplot(all_data)
        plt.plot([1,2,3,4,5],all_mean,'-o',linewidth=2)
        # plt.plot([1,2,3,4,5],all_mean,linewidth=2)
        plt.xticks([0,1,2,3,4,5],
            ['','9', '10','11','12','13'])
        plt.xlabel("Number of Robots", fontdict={
            'fontsize': 18})
        plt.ylabel("Convergence time", fontdict={
            'fontsize': 18})
        plt.xlim([0,6])
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.tight_layout(pad=0.1)
        plt.show()
            
        
    def extract_states(self, keyname: str, states_data: dict):
        extracted_state = []
        for state_data in states_data:
            state = {keyname: state_data[keyname]}
            state.update({"ts": state_data["ts"]})
            extracted_state.append(state)
        return extracted_state
    
    def extract_distribution_evenness(self,all_herd_states, all_shepherd_states):
        herd_distribution_eveness = np.zeros(len(all_shepherd_states))
        herd_distribution_mean = np.zeros(len(all_shepherd_states))
        for indx, shepherd_state in enumerate(all_shepherd_states):
            states = shepherd_state["shepherd"]
            herd_states = all_herd_states[indx]["herd"]
            d_mean = 0
            all_d_to_i = np.zeros((states.shape[0]))
            for i in range(states.shape[0]):
                d_to_i = np.linalg.norm(
                    herd_states[:, :2] - states[i, :2], axis=1)
                d_to_i = np.delete(d_to_i, np.where(d_to_i == 0.0))
                min_d_to_i = d_to_i.min()
                all_d_to_i[i] = min_d_to_i
                d_mean += min_d_to_i
            d_mean = d_mean / states.shape[0]
            herd_distribution_mean[indx] = d_mean
            di_to_d_mean = np.sum(
                np.abs(all_d_to_i - d_mean))**2 / states.shape[0]
            herd_distribution_eveness[indx] = di_to_d_mean

        shepherd_distribution_eveness = np.zeros(len(all_shepherd_states))
        shepherd_distribution_mean = np.zeros(len(all_shepherd_states))
        for indx, shepherd_state in enumerate(all_shepherd_states):
            states = shepherd_state["shepherd"]
            shepherd_states = states.copy()
            d_mean = 0
            all_d_to_i = np.zeros((states.shape[0]))
            for i in range(states.shape[0]):
                d_to_i = np.linalg.norm(
                    shepherd_states[:, :2] - states[i, :2], axis=1)
                d_to_i = np.delete(d_to_i, np.where(d_to_i == 0.0))
                smallest_1, smallest_2 = np.partition(d_to_i, 1)[0:2]
                dr_to_nearest_edge = smallest_1 + smallest_2
                all_d_to_i[i] = dr_to_nearest_edge
                d_mean += dr_to_nearest_edge
            d_mean = d_mean / states.shape[0]
            shepherd_distribution_mean[indx] = d_mean
            di_to_d_mean = np.sum(
                np.abs(all_d_to_i - d_mean))**2 / states.shape[0]
            shepherd_distribution_eveness[indx] = di_to_d_mean

        return herd_distribution_eveness, shepherd_distribution_eveness
    
    def extract_isolation_rate(self, herd_states, shepherd_states):
        isolation_rate = np.zeros((len(shepherd_states)))
        shepherd_centroids = np.zeros((len(shepherd_states), 2))
        d_r_furthest_to_shepherd_centroid = np.zeros(
            (len(shepherd_states)))
        for indx, shepherd_state in enumerate(shepherd_states):
            states = shepherd_state["shepherd"]
            centroid = np.sum(states[:, :2], axis=0) / states.shape[0]
            shepherd_centroids[indx, :] = centroid
            d_to_herd_centroid = np.linalg.norm(
                states[:, :2] - centroid, axis=1)
            d_r_furthest_to_shepherd_centroid[indx] = d_to_herd_centroid.max()

        first_zero = 0
        captured_zero = False
        for indx, herd_state in enumerate(herd_states):
            states = herd_state["herd"]
            d_to_shepherd_centroid = np.linalg.norm(
                states[:, :2] - shepherd_centroids[indx], axis=1)
            isolate_factor = d_to_shepherd_centroid >= d_r_furthest_to_shepherd_centroid[indx]
            rate = np.sum(isolate_factor.astype(np.int8)) / \
                isolate_factor.shape[0]
            isolation_rate[indx] = rate
            if rate == 0.0 and not captured_zero:
                first_zero = indx
                captured_zero = True
        return isolation_rate, first_zero
    
if __name__ == '__main__':
    analyser = DataAnalyser()