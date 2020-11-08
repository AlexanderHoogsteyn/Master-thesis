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

"""
Toolset for identifying phase connectivity of customers in a distribution network.
Author: Alexander Hoogsteyn
Date:   16-09-2020
"""


class Feeder(object):
    """
    A Featureset object contains all the data that you want to use to perform the clustering. The path attribute is
    used to specify the folder which contains the JSON files. The include_empty_feeders is used to specify whether
    you want to include feeders that doe not have any devices i.e. customers connected to it. The other attributes
    make it possible to specify which features to include: - include_n_customers: Number of devices i.e. customers
    connected to a feeder - include_total_length: Total conductor length in the feeder - include_main_path: Longest
    path in the network between a device and the head of the feeder - include_avg_cons: The average active yearly
    energy consumption of the customers on a feeder - include_avg_reactive_cons: Idem for reactive energy consumption
    - include_n_PV: Number of PV installations on the network - include_total_impedance: The impedance between a
    customer and the head of the feeder summed up for all customers - include_average_length: The average path length
    between a customer and the head of the feeder - include_average_impedance: The average impedance between a
    customer and the head of the feeder The object will store the features in  a numpy array as well as some metadata
    such as list of the features used and the ID's of the feeders.
    """

    def __init__(self, path_data='C:/Users/AlexH/OneDrive/Documenten/Julia/Master-thesis/POLA_data/',
                 path_topology='C:/Users/AlexH/OneDrive/Documenten/Julia/Master-thesis/POLA/',
                 feederID='65019_74469', include_three_phase=False, measurement_error=0.0):
        """
        Initialize the featureset by reading out the data from JSON files in the specified directory
        """
        features = []
        if os.path.exists(path_data):
            self._path_data = path_data
        else:
            raise NameError("Data path doesn't exist")
        if os.path.exists(path_topology):
            self._path_topology = path_topology
        else:
            raise NameError("Topology path doesn't exist")

        # list = ["Number of customers","Yearly consumption per customer (kWh)","Yearly reactive consumption per
        # customer (kWh)","Number of PV installations", \ "Total conductor length (km)","Main path length (km)",
        # "Average length to customer (km)", "Total line impedance (Ohm)","Average path impedance (Ohm)"] includes =
        # [include_n_customer, include_avg_cons, include_avg_reactive_cons, include_n_PV, \ include_total_length,
        # include_main_path, include_average_length, include_total_impedance, include_average_impedance]
        # self._feature_list = [list[i] for i in error_range(len(list)) if includes[i]] cycle through all the json files
        # 3 phase loads are added as 3 single phase loads, their Id's are seperately stored as well, the first one is choses
        # as reference for the correlation algorithms
        configuration_file = self._path_topology + feederID + '_configuration.json'
        with open(configuration_file) as current_file:
            config_data = json.load(current_file)
        self._id = config_data['gridConfig']['id']
        self._transfo_id = config_data['gridConfig']['trafoId']
        devices_path = config_data['gridConfig']['devices_file']
        # branches_path = config_data['gridConfig']['branches_file']
        self._nb_customers = config_data['gridConfig']['totalNrOfEANS']

        self._phase_labels = []
        self._device_IDs = []
        self._3phase_IDs = []
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
                for phase in device_phases:
                    if phase == 1:
                        voltage_features.append(voltage_data[str(busID)]["phase_A"])
                        load_features.append(load_data[str(deviceID)]["phase_A"])
                    elif phase == 2:
                        voltage_features.append(voltage_data[str(busID)]["phase_B"])
                        load_features.append(load_data[str(deviceID)]["phase_B"])
                    elif phase == 3:
                        voltage_features.append(voltage_data[str(busID)]["phase_C"])
                        load_features.append(load_data[str(deviceID)]["phase_C"])
                    else:
                        raise NameError("Unkown phase connection")
                if len(device_phases) == 3:
                    self._3phase_IDs += [deviceID]
                    self._device_IDs += [deviceID, deviceID, deviceID]
                else:
                    self._device_IDs += [deviceID]
                self._phase_labels += device_phases
        noise = np.random.normal(0, measurement_error, [np.size(voltage_features, 0), np.size(voltage_features, 1)])
        self._voltage_features = np.array(voltage_features) + noise
        self._load_features = np.array(load_features)
        self._phase_labels = np.array(self._phase_labels)

    def get_voltage_features(self):
        """
        Method to obtain the features as a numpy 2D array, each column contains a feature.
        """
        return self._voltage_features

    def set_voltage_features(self, data):
        self._voltage_features = data

    def get_load_features(self):
        """
        Method to obtain the features as a numpy 2D array, each column contains a feature.
        """
        return self._load_features

    def set_load_features(self, data):
        self._load_features = data

    def get_IDs(self):
        """
        Method to obtain a numpy array of the feeders used, the indeces will correspond to the indeces on the rows
        obtained using get_features(), get_feature() or Clusters.get_clusters()
        """
        return self._device_IDs

    def get_phase_labels(self):
        """
        Method to obtain a list of the features used, the order of which will correspond to the order of the columns in
        get_features()
        """
        return self._phase_labels

    def get_nb_customers(self):
        return self._nb_customers

    def hierarchal_clustering(self, n_clusters=3, normalized=True, criterion='avg_silhouette'):
        """
        Method that returns a clustering object obtained by performing hierarchal clustering of the specified featureset
        By default the features will be normalized first. By scaling the features to have a mean of 0 and unit variance.
        (More info: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
        """
        if normalized:
            scaler = StandardScaler()
            data = scaler.fit_transform(self.get_voltage_features())
        else:
            data = self.get_voltage_features()
        labels = AgglomerativeClustering(n_clusters).fit(data).labels_
        if criterion == 'global_silhouette':
            score = global_silhouette_criterion(data, labels)
        if criterion == 'avg_silhouette':
            score = silhouette_score(data, labels)
        return Cluster(labels, 'hierarchal clustering', normalized, 1, criterion, score)

    def k_means_clustering(self, n_clusters=3, normalized=True, n_repeats=1, criterion='avg_silhouette'):
        """
        Method that returns a clustering object obtained by performing K-means++ on the specified featureset.
        A number of repetitions can be specified, the best result according to the specified criterion will be returned
        By default the features will be normalized first. By scaling the features to have a mean of 0 and unit variance.
        (More info: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
        """
        if normalized == True:
            scaler = StandardScaler()
            data = scaler.fit_transform(self.get_voltage_features())
        else:
            data = self.get_voltage_features()

        if criterion == 'avg_silhouette':
            best_cluster_labels = np.zeros(np.size(data, 0))
            score = -1
            for i in range(0, n_repeats):
                i_cluster_labels = KMeans(n_clusters).fit(data).labels_
                i_silhouette_avg = silhouette_score(data, i_cluster_labels)
                if i_silhouette_avg > score:
                    score = i_silhouette_avg
                    best_cluster_labels = i_cluster_labels
        if criterion == 'global_silhouette':
            best_cluster_labels = np.zeros(np.size(data, 0))
            score = -1
            for i in range(0, n_repeats):
                i_cluster_labels = KMeans(n_clusters).fit(data).labels_
                i_silhouette_global = global_silhouette_criterion(data, i_cluster_labels)
                if i_silhouette_global > score:
                    score = i_silhouette_global
                    best_cluster_labels = i_cluster_labels
        return Cluster(best_cluster_labels, 'k-means++', normalized, n_repeats, criterion, score)

    def k_medoids_clustering(self, n_clusters=3, normalized=True, n_repeats=1, criterion='global_silhouette'):
        """
        Method that returns a clustering object obtained by performing K-medoids++ on the specified featureset.
        A number of repetitions can be specified, the best result according to the specified criterion will be returned
        By default the features will be normalized first. By scaling the features to have a mean of 0 and unit variance.
        (More info: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
        """
        if normalized == True:
            scaler = StandardScaler()
            data = scaler.fit_transform(self.get_voltage_features())
        else:
            data = self.get_voltage_features()

        if criterion == 'avg_silhouette':
            best_cluster_labels = np.zeros(np.size(data, 0))
            score = -1
            for i in range(0, n_repeats):
                i_cluster_labels = KMedoids(n_clusters, init='k-medoids++').fit(data).labels_
                i_silhouette_avg = silhouette_score(data, i_cluster_labels)
                if i_silhouette_avg > score:
                    score = i_silhouette_avg
                    best_cluster_labels = i_cluster_labels
        if criterion == 'global_silhouette':
            best_cluster_labels = np.zeros(np.size(data, 0))
            score = -1
            for i in range(0, n_repeats):
                i_cluster_labels = KMedoids(n_clusters, init='k-medoids++').fit(data).labels_
                i_silhouette_global = global_silhouette_criterion(data, i_cluster_labels)
                if i_silhouette_global > score:
                    score = i_silhouette_global
                    best_cluster_labels = i_cluster_labels
        return Cluster(best_cluster_labels, 'k-medoids++', normalized, n_repeats, criterion, score)

    def gaussian_mixture_model(self, n_clusters=3, normalized=True, n_repeats=1):
        """
        Method that returns a clustering object obtained by performing K-means++ on the specified featureset.
        A number of repetitions can be specified, the best result according to the average silhouette score will be
        returned
        By default the features will be normalized first. By scaling the features to have a mean of 0 and unit variance.
        (More info: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
        """
        if normalized == True:
            scaler = StandardScaler()
            data = scaler.fit_transform(self.get_voltage_features())
        else:
            data = self.get_voltage_features()

        best_cluster_labels = np.zeros(np.size(data, 0))
        score = -1
        reg_values = np.linspace(0.001, .1, num=n_repeats)
        for i in range(0, n_repeats):
            i_cluster_labels = np.array(
                GaussianMixture(n_components=n_clusters, reg_covar=reg_values[i]).fit_predict(data))
            i_silhouette_avg = silhouette_score(data, i_cluster_labels)
            if i_silhouette_avg > score:
                score = i_silhouette_avg
                best_cluster_labels = i_cluster_labels
        return Cluster(best_cluster_labels, 'Gaussian mixture model', normalized, n_repeats, 'avg_silhouette', score)

    def get_reference_3phase_customer(self):
        id_ = np.array(self._device_IDs)
        try:
            id_3 = self._3phase_IDs[0]
        except IndexError:
            raise ValueError("No 3 phase reference found")
        else:
            profiles = self.get_voltage_features()
            profiles = profiles[id_ == id_3]
            labels = self.get_phase_labels()
            labels = labels[id_ == id_3]
        return labels, profiles

    def voltage_correlation(self):
        labels, profiles = self.get_reference_3phase_customer()
        phase_labels = []
        scores = []
        for device in self.get_voltage_features():
            corr = 0
            label = np.nan
            for phase in range(0, 3):
                n_corr = np.correlate(device, profiles[phase])
                if n_corr > corr:
                    corr = n_corr
                    label = labels[phase]
            phase_labels += [label]
            scores += [corr]
        return PhaseIdentification(phase_labels, "voltage_correlation", n_repeats=1, criterion='None', score=scores)


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
            color = plt.cm.viridis(float(i) / (float(self.get_nb_customers()) - 1.0))
            plt.plot(x, line, color=color, alpha=0.85)
            i = i + 1
        plt.xlabel("time step (30 min intervals")
        plt.ylabel(ylabel)
        # plt.title(Cluster.get_algorithm() + " with n_clusters = %d" % Cluster.get_n_clusters() + Cluster.get_repeats())
        plt.show()

    def plot_voltages(self, ylabel="Voltage (pu)", length=48):
        return self.plot_data(self.get_voltage_features(), ylabel, length)

    def plot_load_profiles(self, ylabel="Power (kW)", length=48):
        return self.plot_data(self.get_load_features() * 500, ylabel, length)

    def change_data_representation(self, representation="delta", data="voltage", inplace=True):
        if data == "voltage":
            original_data = self.get_voltage_features()
        elif data == "load":
            original_data = self.get_load_features()
        else:
            return print("enter voltage or load as data")

        new_data = []
        if representation == "delta" or representation == "binary":
            for row in original_data:
                new_row = [0] * len(row)
                for i in range(1, len(row)):
                    new_row[i] = row[i] - row[i - 1]
                new_data.append(new_row)
            new_data = np.array(new_data)
        if representation == "binary":
            new_data[new_data > 0] = 1
            new_data[new_data < 0] = -1
        if inplace:
            if data == "voltage":
                self.set_voltage_features(np.array(new_data))
            if data == "load":
                self.set_load_features(np.array(new_data))
        else:
            new_self = copy.deepcopy(self)
            if data == "voltage":
                new_self.set_voltage_features(np.array(new_data))
            if data == "load":
                new_self.set_load_features(np.array(new_data))
            return new_self

    def add_noise(self, error=0, data="voltage",inplace=True):
        if inplace:
            if data == "voltage":
                voltage_features = self.get_voltage_features()
                noise = np.random.normal(0, error, [np.size(voltage_features, 0), np.size(voltage_features, 1)])
                self.set_voltage_features(voltage_features + noise)
            if data == "load":
                load_features = self.get_load_features()
                noise = np.random.normal(0, error, [np.size(load_features, 0), np.size(load_features, 1)])
                self.set_load_features(load_features + noise)
        else:
            if data == "voltage":
                voltage_features = self.get_voltage_features()
                noise = np.random.normal(0, error, [np.size(voltage_features, 0), np.size(voltage_features, 1)])
                new_self = copy.deepcopy(self)
                new_self.set_voltage_features(voltage_features + noise)
            if data == "load":
                load_features = self.get_load_features()
                noise = np.random.normal(0, error, [np.size(load_features, 0), np.size(load_features, 1)])
                new_self = copy.deepcopy(self)
                new_self.set_load_features(load_features + noise)
            return new_self


class PhaseIdentification(object):
    """
    A phaseIdentification object is formed by performing one of the phase identification methods on the feeder object
    It contains most notably an array with the found phase labels by the method
    """
    def __init__(self, phase_labels, algorithm, n_repeats=1, criterion='global_silhouette', score=np.nan):
        self._phase_labels = phase_labels
        self._algorithm = algorithm
        self._n_clusters = 3
        self._n_repeats = n_repeats
        self._criterion = criterion
        self._score = score

    def get_phase_labels(self):
        return self._phase_labels

    def set_phase_labels(self, labels):
        self._phase_labels = labels

    def accuracy(self, Feeder):
        correct_labels = Feeder.get_phase_labels()
        labels = self.get_phase_labels()
        if len(labels) != len(correct_labels):
            raise IndexError("Phase labels not of same length")
        c = 0.0
        for i in range(0, len(labels)):
            if labels[i] == correct_labels[i]:
                c = c + 1.0
        return c / len(labels)

    def find_wrong_IDs(self, Feeder):
        correct_labels = Feeder.get_phase_labels()
        labels = self.get_phase_labels()
        id_s = Feeder.get_IDs()
        wrong_ids = []
        if len(labels) != len(correct_labels):
            raise IndexError("Phase labels not of same length")
        for i in range(0, len(labels)):
            if labels[i] != correct_labels[i]:
                wrong_ids += [id_s[i]]
        return np.array(wrong_ids)

    def match_labels(self, Feeder):
        best_labels = self.get_phase_labels()
        best_acc = 0.0
        for i in range(0, 7):
            acc = self.accuracy(Feeder)
            labels = self.get_phase_labels()
            if acc > best_acc:
                best_acc = acc
                best_labels = labels
            if i == 3:
                for j in range(0, len(labels)):
                    if labels[j] == 1:
                        labels[j] = 2
                    elif labels[j] == 2:
                        labels[j] = 1
                self.set_phase_labels(np.array(labels))
            else:
                self.set_phase_labels(list(map(lambda x: x % 3 + 1, labels)))
        self.set_phase_labels(np.array(best_labels))

    def plot_voltages(Feeder, PhaseIdentification, x_axis=None, y_axis=None):
        """
        Makes a 2D plot of the resulting clusters. You need to specify the Feeder object which contains all the used data
        as well as the Cluster object which you obtained by performing one on the clustering algorithm methods
        on the Feeder.
        It can be chosen what is plotted on the x and y axis by specifying the name of a feature. This has to be the specific
        string corresponding to that feature such as "Yearly consumption per customer (kWh)" (These can be found using
        Feeder.get_feature_list() ).
        """
        cluster_labels = PhaseIdentification.get_phase_labels()
        voltage_data = Feeder.get_voltage_features()
        plt.figure(figsize=(8, 6))
        markers = ["s", "o", "D", ">", "<", "v", "+"]
        x = np.arange(0, 48)
        for i in range(0, Cluster.get_n_clusters()):
            color = plt.cm.viridis(float(i) / (float(Cluster.get_n_clusters()) - 1.0))
            for line in voltage_data:
                plt.plot(x, line, color=color, alpha=0.85)
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.title(Cluster.get_algorithm() + " with n_clusters = %d" % Cluster.get_n_clusters() + Cluster.get_repeats())
        plt.show()


class Cluster(PhaseIdentification):
    """
    Special case of  PhaseIdentification class when result is obtained by performing a clustering method.
    A cluster object contains all info on the result obtained after performing a clustering algorithm. Most notably the
    labels to identify which cluster a feeder is allocated to. Besides that, the object contains some metadata about
    the number of clusters, which algorithm was used, the score of the result according to the specified criterion.
    """

    def __init__(self, clusters, algorithm, normalized=False, n_repeats=1, criterion='global_silhouette', score=np.nan):
        self._phase_labels = clusters + 1
        self._algorithm = algorithm
        self._normalized = normalized
        self._n_clusters = np.max(clusters) + 1
        self._n_repeats = n_repeats
        self._criterion = criterion
        self._score = score

    def get_algorithm(self):
        return self._algorithm

    def get_normalisation(self):
        if self._normalized == True:
            return 'normalized'
        else:
            return 'not normalized'

    def get_n_repeats(self):
        return self._n_repeats

    def get_repeats(self):
        if self.get_n_repeats() == 1:
            return ''
        else:
            return ' repeated %d times' % self.get_n_repeats()

    def get_criterion(self):
        return self._criterion

    def is_normalised(self):
        return self._normalized

    def get_n_clusters(self):
        return self._n_clusters

    def get_score(self):
        return self._score


def silhouette_analysis(Feeder, Cluster):
    """
    Makes a silhouette analysis of the resulting clusters (more info: https://en.wikipedia.org/wiki/Silhouette_(clustering) ).
    You need to specify the Feeder object which contains all the used data as well as the Cluster object
    which you obtained by performing one on the clustering algorithm methods on the Feeder.
    """
    features = Feeder.get_features()
    cluster_labels = Cluster.get_clusters()
    n_clusters = Cluster.get_n_clusters()
    feature_list = Feeder.get_feature_list()
    plt.figure(figsize=(22, 10))
    plt.set_xlim([-0.1, 1])
    plt.set_ylim([0, len(features) + (n_clusters + 1) * 10])

    if Cluster.is_normalised():
        scaler = StandardScaler()
        features_normalised = scaler.fit_transform(features)
        silhouette_avg = silhouette_score(features_normalised, cluster_labels)
        silhouette_global = global_silhouette_criterion(features_normalised, cluster_labels)
        sample_silhouette_values = silhouette_samples(features_normalised, cluster_labels)
    else:
        silhouette_avg = silhouette_score(features, cluster_labels)
        silhouette_global = global_silhouette_criterion(features, cluster_labels)
        sample_silhouette_values = silhouette_samples(features, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = plt.cm.viridis(float(i) / (n_clusters - 1))

        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i + 1))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="grey", linestyle="--", label='average silhouette coef %f3' % silhouette_avg)
    ax1.axvline(x=silhouette_global, color="grey", linestyle="-.",
                label='global silhouette coef %f3' % silhouette_global)
    ax1.legend(loc='upper right')

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.show()


def compare_algorithms(Feeder, criterion, n=1, range=range(2, 25)):
    """
    Makes a graph that compares the 4 algorithms against each other according to their average silhouette coefficient.
    A featureset needs to be specified to perform the analysis on.
    A error_range of number of clusters must be specified.
    A number of repetitions can be specified, K-means++, K-medoids++ and GMM will then be repeated n times and the best
    result is kept. Hierarchal clustering is only performed once because its outcome is not stochastical.
    """
    results = {'Hierarchal': dict(), 'K-means++': dict(), 'K-medoids++': dict(), 'GMM': dict()}
    scores = np.zeros([4, len(range)])
    for i in range:
        results['Hierarchal'][i] = Feeder.hierarchal_clustering(n_clusters=i)
        results['K-means++'][i] = Feeder.k_means_clustering(n_clusters=i, n_repeats=n)
        results['K-medoids++'][i] = Feeder.k_medoids_clustering(n_clusters=i, n_repeats=n)
        results['GMM'][i] = Feeder.gaussian_mixture_model(n_clusters=i, n_repeats=n)
        if criterion == "accuracy":
            results['Hierarchal'][i].match_labels(Feeder)
            results['K-means++'][i].match_labels(Feeder)
            results['K-medoids++'][i].match_labels(Feeder)
            results['GMM'][i].match_labels(Feeder)
            scores[0, i - range[0]] = results['Hierarchal'][i].accuracy(Feeder)
            scores[1, i - range[0]] = results['K-means++'][i].accuracy(Feeder)
            scores[2, i - range[0]] = results['K-medoids++'][i].accuracy(Feeder)
            scores[3, i - range[0]] = results['GMM'][i].accuracy(Feeder)
        else:
            scores[0, i - range[0]] = results['Hierarchal'][i].get_score()
            scores[1, i - range[0]] = results['K-means++'][i].get_score()
            scores[2, i - range[0]] = results['K-medoids++'][i].get_score()
            scores[3, i - range[0]] = results['GMM'][i].get_score()
        print(i, " out of ", range[-1], " complete")
    plt.figure(figsize=(12, 10))
    plt.plot(range, scores[0, :], label='Hierarchal')
    plt.plot(range, scores[1, :], label='K-means++')
    plt.plot(range, scores[2, :], label='K-medoids++')
    plt.plot(range, scores[3, :], label='GMM')
    plt.legend()
    plt.xlabel("number of clusters K")
    if criterion == "accuracy":
        plt.ylabel("Accuracy (%)")
    else:
        plt.ylabel("average silhouette coefficient")
    plt.show()
    return results, scores


def variance_ratio_criterion():
    raise NotImplementedError


def global_silhouette_criterion(features, cluster_labels):
    nb_clusters = np.max(cluster_labels) + 1
    scores = silhouette_samples(features, cluster_labels)
    score = 0
    for i in range(0, nb_clusters):
        score += scores[cluster_labels == i].mean()
    return score / nb_clusters


def consensus_matrix(FeatureSet, n, min_n_clusters, max_n_clusters):
    """
    Function to build the consensus matrix needed for ensemble clustering
    """
    length = len(FeatureSet.get_feature(0))
    similarity_matrix = np.zeros([length, length])
    for i in range(0, n):
        cluster_labels = FeatureSet.k_means_clustering(
            n_clusters=random.randint(min_n_clusters, max_n_clusters)).get_clusters()
        for j in range(0, length):
            for k in range(j + 1, length):
                if cluster_labels[j] == cluster_labels[k]:
                    similarity_matrix[j][k] += 1
                    similarity_matrix[k][j] += 1
    return similarity_matrix / n


def compare_ensemble_algorithms(FeatureSet, n, range):
    """
    Makes a graph to compare the results of the different ensemble algorithms against each other
    """
    results = {"Average fixed": dict(), "Average varying": dict(), "Single fixed": dict(), "Single varying": dict()}
    scores = np.zeros([4, len(range)])
    mat = consensus_matrix(FeatureSet, n, min(range), max(range))
    scaler = StandardScaler()
    features = FeatureSet.get_features()
    features_normalised = scaler.fit_transform(features)
    for i in range:
        mat_2 = consensus_matrix(FeatureSet, n, i, i)
        cluster_labels = AgglomerativeClustering(n_clusters=i, linkage='average').fit(mat).labels_
        scores[0][i - range[0]] = silhouette_score(features_normalised, cluster_labels)
        results["Average varying"][i] = Cluster(cluster_labels, 'Cluster ensemble', True, 1, 'avg_silhouette',
                                                scores[0][i - range[0]])

        cluster_labels = AgglomerativeClustering(n_clusters=i, linkage='average').fit(mat_2).labels_
        scores[1][i - range[0]] = silhouette_score(features_normalised, cluster_labels)
        results["Average fixed"][i] = Cluster(cluster_labels, 'Cluster ensemble', True, 1, 'avg_silhouette',
                                              scores[1][i - range[0]])

        cluster_labels = AgglomerativeClustering(n_clusters=i, linkage='single').fit(mat).labels_
        scores[2][i - range[0]] = silhouette_score(features_normalised, cluster_labels)
        results["Single varying"][i] = Cluster(cluster_labels, 'Cluster ensemble', True, 1, 'avg_silhouette',
                                               scores[2][i - range[0]])

        cluster_labels = AgglomerativeClustering(n_clusters=i, linkage='single').fit(mat_2).labels_
        scores[3][i - range[0]] = silhouette_score(features_normalised, cluster_labels)
        results["Single fixed"][i] = Cluster(cluster_labels, 'Cluster ensemble', True, 1, 'avg_silhouette',
                                             scores[2][i - range[0]])
    plt.plot(range, scores[0, :], label="average linkage")
    plt.plot(range, scores[2, :], label="single linkage")
    plt.plot(range, scores[1, :], label="average linkage fixed K")
    plt.plot(range, scores[3, :], label="single linkage fixed K")
    plt.legend()
    plt.show()
    return results, scores


def compare_data_representations(feeder, error_range=np.arange(0, 0.01, 0.001), n_repeats=1, algorithm="clustering"):
    results = {"raw": dict(), "delta": dict(), "binary": dict()}
    scores = np.zeros([3, len(error_range)])
    if algorithm == "clustering":
        for i in range(0, len(error_range)):
            feeder_raw = feeder.add_noise(error_range[i], inplace=False)#Feeder(measurement_error=error_range[i])
            feeder_delta = feeder_raw.change_data_representation(representation="delta", inplace=False)
            feeder_binary = feeder_raw.change_data_representation(representation="binary", inplace=False)
            results["raw"][i] = feeder_raw.k_means_clustering(n_repeats=n_repeats)
            results["delta"][i] = feeder_delta.k_means_clustering(n_repeats=n_repeats)
            results["binary"][i] = feeder_binary.k_means_clustering(n_repeats=n_repeats)
            results["raw"][i].match_labels(feeder_raw)
            results["delta"][i].match_labels(feeder_delta)
            results["binary"][i].match_labels(feeder_binary)
            scores[0][i] = results["raw"][i].accuracy(feeder_raw)
            scores[1][i] = results["delta"][i].accuracy(feeder_delta)
            scores[2][i] = results["binary"][i].accuracy(feeder_binary)
            #print(i/len(error_range)*100, "% complete")

    return results, scores


def get_representative_feeders(FeatureSet, Cluster):
    """
    Function that returns a pandas dataframe with a summary of the found clusters and the mean and deviation of the
    features of the feeders in that cluster.
    """
    nb_clusters = Cluster.get_n_clusters()
    cluster_labels = Cluster.get_clusters()
    feature_list = FeatureSet.get_feature_list()
    dict = {'Number of feeders': []}
    for item in feature_list:
        dict[item + " (mean)"] = []
        dict[item + " (std)"] = []
    for i in range(0, nb_clusters):
        dict['Number of feeders'].append(np.count_nonzero(cluster_labels == i))
        for item in feature_list:
            mask = FeatureSet.get_feature(item)[cluster_labels == i]
            dict[item + " (mean)"].append(mask.mean())
            dict[item + " (std)"].append(mask.std())
    return pd.DataFrame(dict)

