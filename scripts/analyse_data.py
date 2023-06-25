# !/usr/bin/python3

import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt


class DataAnalyser(object):
    def __init__(self, data_name=None,
                 load_latest=True):
        self._raw_data = {}
        self._config = {}
        self._states_data = {}

        directory = "data/"
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
            max_idx = np.where(ts_list == ts_list.max())[0][0]
            file = files[max_idx]

            # Unpickle the list
            with open(directory + file, 'rb') as f:
                self._raw_data = pickle.load(f)
            self._config = self._raw_data["configuration"]
            self._states_data = self._raw_data["data"]

        self._herd_states = self.extract_states("herd")
        self._shepherd_states = self.extract_states("shepherd")

    def extract_states(self, keyname: str):
        extracted_state = []
        for state_data in self._states_data:
            state = {keyname: state_data[keyname]}
            state.update({"ts": state_data["ts"]})
            extracted_state.append(state)
        return extracted_state

    def extract_and_plot_distance_to_herd_centroid(self):
        herd_centroids = np.zeros((len(self._herd_states), 2))
        for indx, herd_state in enumerate(self._herd_states):
            states = herd_state["herd"]
            centroid = np.sum(states[:, :2], axis=0) / states.shape[0]
            herd_centroids[indx, :] = centroid
        herd_centroids_norm = np.linalg.norm(herd_centroids, axis=1)

        shepherd_centroids = np.zeros((len(self._shepherd_states), 2))
        for indx, shepherd_state in enumerate(self._shepherd_states):
            states = shepherd_state["shepherd"]
            centroid = np.sum(states[:, :2], axis=0) / states.shape[0]
            shepherd_centroids[indx, :] = centroid
        shepherd_centroids_norm = np.linalg.norm(shepherd_centroids, axis=1)

        d_to_herd_centroid = np.linalg.norm(
            shepherd_centroids - herd_centroids, axis=1)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(d_to_herd_centroid, linewidth=2,snap=True)
        ax.set_title("Distance between robot's centroid and herd's centroid", fontdict={
                     'fontsize': 14})
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Distance")
        ax.set_ylim([0, d_to_herd_centroid.max()])
        ax.set_xlim([0, len(self._herd_states)])

        ax.axvline(x=275, color="gray", linestyle="--", linewidth=1)
        ax.annotate("Edge following\nphase",[80,550])
        ax.annotate("Surrounding\nphase",[375,550])
        x = np.arange(0,len(self._herd_states),1)
        ax.fill_between(x=x,y1=d_to_herd_centroid.max()+100,where=x<275,alpha=0.4)
        ax.annotate("",
            xy=(350, 500), xycoords='data',
            xytext=(225, 500), textcoords='data',
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
        
        plt.show()

    def extract_and_plot_distribution_range(self):
        herd_centroids = np.zeros((len(self._herd_states), 2))
        d_s_furthest_to_herd_centroid = np.zeros((len(self._herd_states)))
        for indx, herd_state in enumerate(self._herd_states):
            states = herd_state["herd"]
            centroid = np.sum(states[:, :2], axis=0) / states.shape[0]
            herd_centroids[indx, :] = centroid
            d_to_herd_centroid = np.linalg.norm(states[:,:2] - centroid,axis=1)
            d_s_furthest_to_herd_centroid[indx] = d_to_herd_centroid.max()

        shepherd_centroids = np.zeros((len(self._shepherd_states), 2))
        d_r_furthest_to_herd_centroid = np.zeros((len(self._herd_states)))
        for indx, shepherd_state in enumerate(self._shepherd_states):
            shepherd_state_states = shepherd_state["shepherd"]
            herd_centroid = herd_centroids[indx,:]
            d_to_herd_centroid = np.linalg.norm(shepherd_state_states[:,:2] - herd_centroid,axis=1)
            d_r_furthest_to_herd_centroid[indx] = d_to_herd_centroid.max()

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(d_s_furthest_to_herd_centroid, linewidth=2,label="Herd")
        ax.plot(d_r_furthest_to_herd_centroid, linewidth=2, label="Robot")
        ax.set_title("Distribution Range", fontdict={
                     'fontsize': 14})
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Distance")
        ax.legend()

        ax.set_ylim([0, d_r_furthest_to_herd_centroid.max()+10])
        ax.set_xlim([0, len(self._herd_states)])

        ax.axvline(x=275, color="gray", linestyle="--", linewidth=1)
        ax.annotate("Edge following\nphase",[100,570])
        ax.annotate("Surrounding\nphase",[375,570])
        x = np.arange(0,len(self._herd_states),1)
        ax.fill_between(x=x,y1=d_r_furthest_to_herd_centroid.max()+100,where=x<=275,alpha=0.4)
        ax.annotate("",
            xy=(350, 550), xycoords='data',
            xytext=(225, 550), textcoords='data',
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

        plt.show()

    def extract_and_plot_distribution_eveness(self):
        distribution_eveness = np.zeros(len(self._shepherd_states))
        for indx, shepherd_state in enumerate(self._shepherd_states):
            states = shepherd_state["shepherd"]
            herd_states = self._herd_states[indx]["herd"]
            d_mean = 0
            all_d_to_i = np.zeros((states.shape[0]))
            for i in range(states.shape[0]):
                d_to_i = np.linalg.norm(herd_states[:,:2] - states[i,:2],axis=1)
                d_to_i = np.delete(d_to_i, np.where(d_to_i == 0.0))
                min_d_to_i = d_to_i.min()
                all_d_to_i[i] = min_d_to_i
                d_mean += min_d_to_i
            d_mean = d_mean / states.shape[0]
            di_to_d_mean = np.sum(np.abs(all_d_to_i - d_mean))**2 / states.shape[0]
            distribution_eveness[indx] = di_to_d_mean

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(distribution_eveness, linewidth=2)
        ax.set_title("Distribution Eveness", fontdict={
                     'fontsize': 14})
        ax.set_xlabel("Timestep")

        ax.set_ylim([0, distribution_eveness.max()+1000])
        ax.set_xlim([0, len(self._herd_states)])

        ax.axvline(x=275, color="gray", linestyle="--", linewidth=1)
        ax.annotate("Edge following\nphase",[80,13000])
        ax.annotate("Surrounding\nphase",[350,13000])
        x = np.arange(0,len(self._herd_states),1)
        ax.fill_between(x=x,y1=distribution_eveness.max()+10000,where=x<=275,alpha=0.4)
        ax.annotate("",
            xy=(350, 12000), xycoords='data',
            xytext=(225, 12000), textcoords='data',
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
        plt.show()

    def extract_isolate_rate(self):
        isolation_rate = np.zeros((len(self._shepherd_states)))
        shepherd_centroids = np.zeros((len(self._shepherd_states), 2))
        d_r_furthest_to_shepherd_centroid = np.zeros((len(self._shepherd_states)))
        for indx, shepherd_state in enumerate(self._shepherd_states):
            states = shepherd_state["shepherd"]
            centroid = np.sum(states[:, :2], axis=0) / states.shape[0]
            shepherd_centroids[indx, :] = centroid
            d_to_herd_centroid = np.linalg.norm(states[:,:2] - centroid,axis=1)
            d_r_furthest_to_shepherd_centroid[indx] = d_to_herd_centroid.max()
        
        first_zero = 0
        captured_zero = False
        for indx, herd_state in enumerate(self._herd_states):
            states = herd_state["herd"]
            d_to_shepherd_centroid = np.linalg.norm(states[:,:2] - shepherd_centroids[indx],axis=1)
            isolate_factor = d_to_shepherd_centroid >= d_r_furthest_to_shepherd_centroid[indx]
            rate = np.sum(isolate_factor.astype(np.int8)) / isolate_factor.shape[0]
            isolation_rate[indx] = rate
            if rate == 0.0 and not captured_zero:
                first_zero = indx
                captured_zero = True
        print(first_zero)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(isolation_rate, linewidth=2)
        ax.axvline(x=first_zero, color="gray", linestyle="--", linewidth=1)
        ax.annotate("Edge following\nphase",[5,0.45])
        ax.annotate("Surrounding\nphase",[375,0.45])
        ax.fill_between(x=np.arange(0,len(isolation_rate),1),y1=1.2,y2=-1,where=isolation_rate>0,alpha=0.4)
        ax.set_title("Isolation Rate", fontdict={
                     'fontsize': 14})
        ax.annotate("",
            xy=(350, 0.5), xycoords='data',
            xytext=(225, 0.5), textcoords='data',
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
        ax.set_xlabel("Timestep")
        ax.set_ylabel("")
        ax.set_ylim([-0.01, isolation_rate.max()+0.05])
        ax.set_xlim([0, len(self._herd_states)])
        plt.show()

if __name__ == '__main__':
    analyser = DataAnalyser()
    # analyser.extract_and_plot_distribution_range()
    # analyser.extract_and_plot_distance_to_herd_centroid()
    # analyser.extract_and_plot_distribution_eveness()
    analyser.extract_isolate_rate()
