import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


class AccuracySensitivity():
    def __init__(self,results):
        self.ratio_range = results["ratio_range"]
        self.length_range = results["length_range"]
        self.missing_range = results["missing_range"]
        self.included_feeders = results["included_feeders"]
        self.data = results["data"]

    def merge(self, other):
        """
        Method to merge the data from 2 AccuracySensititvity objects together with different ratio ranges
        """
        self.ratio_range = np.sort(np.concatenate(self.ratio_range, other.ratio_range))
        self.data.update(other.data)
        return self

    def get_index_ratio(self,ratio):
        for i, ratios in enumerate(self.ratio_range):
            if ratio == ratios:
                return i
        raise KeyError("Ratio not in ratio range")

    def visualize_one(self,ratio,feeder):
        plt.figure(figsize=(12, 10), dpi=80)
        y = [str(i) + "%" for i in list(np.arange(100, 0, -10))]
        x = self.length_range
        tup = (ratio,feeder)
        sns.heatmap(self.data[tup], xticklabels=x, yticklabels=y, cmap='RdYlGn', center=0.7,
                    annot=True)

        # Decorations
        plt.title('Percentage of customers that are allocated correctly for ' + str(feeder), fontsize=16)
        plt.xticks(fontsize=12)
        plt.xlabel("Duration (days) that hourly data was collected")
        plt.ylabel("Percentage of customers with smart meter")
        plt.yticks(fontsize=12)
        plt.show()

    def visualize_load_based(self, ratio=0):
        y = [str(i) + "%" for i in list(np.arange(100, 0, -10))]
        x = self.length_range

        i  = self.get_index_ratio(ratio)

        fig, axs = plt.subplots(1, len(self.included_feeders)+1, figsize=(20, 6), dpi=90,
                               gridspec_kw={'width_ratios':[1,1,1,0.08]})

        axs[0].get_shared_y_axes().join(axs[1], axs[2])
        sns.heatmap(100 * self.data[(i, 0)], ax=axs[0], xticklabels=x, yticklabels=y, cmap='RdYlGn',
                    cbar=False, center=70, annot=True)
        sns.heatmap(100 * self.data[(i, 1)], ax=axs[1], xticklabels=x, yticklabels=y, cmap='RdYlGn',
                    cbar=False, center=70, annot=True)
        sns.heatmap(100 * self.data[(i, 2)], ax=axs[2], xticklabels=x, yticklabels=y, cmap='RdYlGn',
                    cbar_ax=axs[3], center=70, annot=True)

        axs[1].set_xlabel("Duration (days) that hourly data was collected", fontsize=12)
        axs[2].set_xlabel("Duration (days) that hourly data was collected", fontsize=12)
        axs[0].set_xlabel("Duration (days) that hourly data was collected", fontsize=12)
        axs[0].set_ylabel("Percentage of customers with smart meter", fontsize=12)
        axs[0].set_title("Case A", fontsize=12)
        axs[1].set_title("Case B", fontsize=12)
        axs[2].set_title("Case C", fontsize=12)


        plt.show(fig)

    def visualize_voltage_based(self):
        self.visualize_load_based(ratio=1.0)


    def visualize_voltage_assisted(self,range):
        fig, axs = plt.subplots(len(range), len(self.included_feeders), figsize=(20, 16), dpi=90, sharex='all',
                                sharey='all')
        # fig.suptitle('Fraction of customers that are allocated correctly')
        for g, feeder in enumerate(self.included_feeders):
            for h, value in enumerate(range):

                y = [str(i) + "%" for i in list(np.arange(100, 0, -10))]
                x = self.length_range
                sns.heatmap(100 * self.data[(h, g)], ax=axs[h, g], xticklabels=x, yticklabels=y, cmap='RdYlGn',
                            cbar=False, center=70,
                            annot=True)

                # Decorations
                axs[h, g].set_title(feeder + ', Voltage assistance ' + str(round(value * 100)) + '%', fontsize=12)
                if h == 2:
                    axs[h, g].set_xlabel("Duration (days) that hourly data was collected", fontsize=12)
                if g == 0:
                    axs[h, g].set_ylabel("Percentage of customers with smart meter", fontsize=12)

        plt.show(fig)

class CorrelationCoeficients():

    def __init__(self,PartialPhaseIdentification):
        self.phase_labels = PartialPhaseIdentification.phase_labels
        self.load_features = PartialPhaseIdentification.load_features
        self._load_features_transfo = PartialPhaseIdentification.load_features_transfo
        self.voltage_features = PartialPhaseIdentification.voltage_features
        self.voltage_features_transfo = PartialPhaseIdentification._voltage_features_transfo

    def pearson_corr_load(self,j,phase):
        phase = phase - 1
        corr = 1.0 / (len(self.phase_labels) - 1) * sum((self.load_features[j] - np.mean(self.load_features[j])) * \
                (self._load_features_transfo[phase] - np.mean(self._load_features_transfo[phase]))) \
                / (np.std(self.load_features[j])*np.std(self._load_features_transfo[phase]))
        return corr

    def pearson_corr_voltage(self,j,phase):
        phase = phase - 1
        corr = 1.0 / (len(self.phase_labels) - 1) * sum((self.voltage_features[j] - np.mean(self.voltage_features[j])) * \
                (self.voltage_features_transfo[phase] - np.mean(self.voltage_features_transfo[phase]))) \
                / (np.std(self.voltage_features[j])*np.std(self.voltage_features_transfo[phase]))
        return corr

    def corr_vector_load(self,phase, reference_phase):
        corr_vector = []
        for j,label in enumerate(self.phase_labels):
            if label == phase:
                corr_vector += [self.pearson_corr_load(j,reference_phase)]
        return np.array(corr_vector)

    def corr_vector_voltage(self,phase, reference_phase):
        corr_vector = []
        for j,label in enumerate(self.phase_labels):
            if label == phase:
                corr_vector += [self.pearson_corr_voltage(j,reference_phase)]
        return np.array(corr_vector)

    def visualize_correlation(self,reference):
        markers = ["s", "o", "D", ">", "<", "v", "+"]
        plt.figure(figsize=(8, 6), dpi=80)
        plt.scatter(self.corr_vector_voltage(1,reference),self.corr_vector_load(1,reference),color='tab:green',marker=markers[0])
        plt.scatter(self.corr_vector_voltage(2,reference),self.corr_vector_load(2,reference),color='tab:red',marker=markers[1])
        plt.scatter(self.corr_vector_voltage(3,reference),self.corr_vector_load(3,reference),color='tab:red',marker=markers[2])
        plt.xlabel("Voltage correlation")
        plt.ylabel("Load correlation")
        plt.show()

    def visualize_correlation_all(self):
        markers = ["s", "o", "D", ">", "<", "v", "+"]
        plt.figure(figsize=(8, 6), dpi=80)
        plt.scatter(self.corr_vector_voltage(1,1),self.corr_vector_load(1,1),color='tab:green',marker=markers[0])
        plt.scatter(self.corr_vector_voltage(2,1),self.corr_vector_load(2,1),color='tab:red',marker=markers[1])
        plt.scatter(self.corr_vector_voltage(3,1),self.corr_vector_load(3,1),color='tab:red',marker=markers[2])
        plt.scatter(self.corr_vector_voltage(1,2),self.corr_vector_load(1,2),color='tab:red',marker=markers[0])
        plt.scatter(self.corr_vector_voltage(2,2),self.corr_vector_load(2,2),color='tab:green',marker=markers[1])
        plt.scatter(self.corr_vector_voltage(3,2),self.corr_vector_load(3,2),color='tab:red',marker=markers[2])
        plt.scatter(self.corr_vector_voltage(1,3),self.corr_vector_load(1,3),color='tab:red',marker=markers[0])
        plt.scatter(self.corr_vector_voltage(2,3),self.corr_vector_load(2,3),color='tab:red',marker=markers[1])
        plt.scatter(self.corr_vector_voltage(3,3),self.corr_vector_load(3,3),color='tab:green',marker=markers[2])
        plt.xlabel("Voltage correlation", fontsize=12)
        plt.ylabel("Load correlation", fontsize=12)
        plt.show()
