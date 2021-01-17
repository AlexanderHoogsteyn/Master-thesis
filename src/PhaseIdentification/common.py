import json
import random
import numpy as np
import pandas as pd
import os
import copy
import glob
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt


class Feeder(object):
    """
    A Feeder object contains all the data (voltage + load) that is needed to perform the clustering. The path_topology attribute is
    used to specify the folder which contains the JSON files. The object will store the features in  a numpy array as well as some metadata
    such as list of the features used and the ID's of the feeders.
    """

    def __init__(self, feederID='65019_74469', include_three_phase=False, measurement_error=0.0,length=24):
        """
        Initialize the feeder object by reading out the data from JSON files in the specified directory
        """
        features = []
        dir = os.path.dirname(os.path.realpath(__file__))
        self._path_data = os.path.join(dir, "../../data/POLA_data/")
        self._path_topology = os.path.join(dir, "../../data/POLA/")
        self.length = length


        configuration_file = self._path_topology + feederID + '_configuration.json'
        with open(configuration_file) as current_file:
            config_data = json.load(current_file)
        self.id = config_data['gridConfig']['id']
        self.transfo_id = config_data['gridConfig']['trafoId']
        devices_path = config_data['gridConfig']['devices_file']
        # branches_path = config_data['gridConfig']['branches_file']
        self.nb_customers = config_data['gridConfig']['totalNrOfEANS']

        self.phase_labels = []
        self.device_IDs = []
        self.multiphase_IDs = []
        voltage_features = []
        load_features = []

        with open(os.path.join(os.path.dirname(self._path_topology), feederID + "_devices.json")) as devices_file:
            devices_data = json.load(devices_file)
        with open(os.path.join(os.path.dirname(self._path_data), feederID + "_voltage_data.json")) as voltage_file:
            voltage_data = json.load(voltage_file)
        with open(os.path.join(os.path.dirname(self._path_data), feederID + "_load_data.json")) as load_file:
            load_data = json.load(load_file)

        for device in devices_data['LVcustomers']:
            deviceID = device.get("deviceId")
            busID = device.get("busId")
            device_phases = device.get("phases")
            if include_three_phase or len(device_phases) == 1:
                #print("device: ", deviceID, " bus: ", busID, " phase: ", device_phases)
                for phase in device_phases:
                    if phase == 1:
                        voltage_features.append(voltage_data[str(busID)]["phase_A"][0:length])
                        load_features.append(load_data[str(deviceID)]["phase_A"][0:length])
                    elif phase == 2:
                        voltage_features.append(voltage_data[str(busID)]["phase_B"][0:length])
                        load_features.append(load_data[str(deviceID)]["phase_B"][0:length])
                    elif phase == 3:
                        voltage_features.append(voltage_data[str(busID)]["phase_C"][0:length])
                        load_features.append(load_data[str(deviceID)]["phase_C"][0:length])
                    else:
                        raise NameError("Unkown phase connection")
                if len(device_phases) == 3:
                    self.multiphase_IDs += [deviceID]
                    self.device_IDs += [deviceID, deviceID, deviceID]
                else:
                    self.device_IDs += [deviceID]
                self.phase_labels += device_phases

        self._voltage_features_transfo = np.zeros([3, length])
        self._voltage_features_transfo[0] = voltage_data["transfo"]["phase_A"][0:length]
        self._voltage_features_transfo[1] = voltage_data["transfo"]["phase_B"][0:length]
        self._voltage_features_transfo[2] = voltage_data["transfo"]["phase_C"][0:length]

        self.load_features_transfo = np.zeros([3, length])
        self.load_features_transfo[0] = load_data["transfo"]["phase_A"][0:length]
        self.load_features_transfo[1] = load_data["transfo"]["phase_B"][0:length]
        self.load_features_transfo[2] = load_data["transfo"]["phase_C"][0:length]

        self._load_features_transfo = np.zeros([3, length])
        self._load_features_transfo[0] = load_data["transfo"]["phase_A"][0:length]
        self._load_features_transfo[1] = load_data["transfo"]["phase_B"][0:length]
        self._load_features_transfo[2] = load_data["transfo"]["phase_C"][0:length]

        noise = np.random.normal(0, measurement_error, [np.size(voltage_features, 0), np.size(voltage_features, 1)])
        self.voltage_features = np.array(voltage_features) + noise
        self.load_features = np.array(load_features)
        self.phase_labels = np.array(self.phase_labels)


    def plot_data(self, data, ylabel="(pu)", length=48):
        """
        Makes a 2D plot of the resulting clusters. You need to specify the Feeder object which contains all the used data
        as well as the Cluster object which you obtained by performing one on the clustering algorithm methods
        on the Feeder.
        It can be chosen what is plotted on the x and y axis by specifying the name of a feature. This has to be the specific
        string corresponding to that feature such as "Yearly consumption per customer (kWh)" (These can be found using
        Feeder.get_feature_list() ).
        """
        plt.figure(figsize=(8, 6))
        markers = ["s", "o", "D", ">", "<", "v", "+"]
        x = np.arange(0, length)
        i = 0
        for line in data:
            line = line[0:length]
            color = plt.cm.viridis(float(i) / (float(self.nb_customers) - 1.0))
            plt.plot(x, line, color=color, alpha=0.85)
            i = i + 1
        plt.xlabel("time step (30 min intervals")
        plt.ylabel(ylabel)
        plt.show()

    def plot_voltages(self, ylabel="Voltage (pu)", length=48):
        return self.plot_data(self.voltage_features, ylabel, length)

    def plot_load_profiles(self, ylabel="Power (kW)", length=48):
        return self.plot_data(self.load_features * 500, ylabel, length)

    def change_data_representation(self, representation="delta", data="voltage", inplace=True):
        if data == "voltage":
            original_data = self.voltage_features
            original_transfo_data = self._voltage_features_transfo
        elif data == "load":
            original_data = self.load_features
            original_transfo_data = self._voltage_features_transfo
        else:
            return print("enter voltage or load as data")

        new_data = []
        new_transfo_data = []
        if representation == "delta" or representation == "binary":
            for row in original_data:
                new_row = [0] * len(row)
                for i in range(1, len(row)):
                    new_row[i] = row[i] - row[i - 1]
                new_data.append(new_row)
            new_data = np.array(new_data)

            for row in original_transfo_data:
                new_row = [0] * len(row)
                for i in range(1, len(row)):
                    new_row[i] = row[i] - row[i - 1]
                new_transfo_data.append(new_row)
            new_transfo_data = np.array(new_transfo_data)

        if representation == "binary":
            new_data[new_data > 0] = 1
            new_data[new_data < 0] = -1
            new_transfo_data[new_transfo_data > 0] = 1
            new_transfo_data[new_transfo_data < 0] = -1
        if inplace:
            if data == "voltage":
                self.voltage_features = np.array(new_data)
                self._voltage_features_transfo = np.array(new_transfo_data)
            if data == "load":
                self.load_features = np.array(new_data)
                self._load_features_transfo = np.array(new_transfo_data)
        else:
            new_self = copy.deepcopy(self)
            if data == "voltage":
                new_self.voltage_features = np.array(new_data)
                new_self._voltage_features_transfo = np.array(new_transfo_data)

            if data == "load":
                new_self.load_features = np.array(new_data)
                new_self._voltage_features_transfo = np.array(new_transfo_data)
            return new_self

    def truncate_voltages(self):
        vf = self.voltage_features
        vf = vf*230     #change from pu to V
        vf = np.trunc(vf)
        self.voltage_features = vf



