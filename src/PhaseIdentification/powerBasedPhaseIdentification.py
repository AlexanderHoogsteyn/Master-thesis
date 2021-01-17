from PhaseIdentification.common import *
import numpy as np


class PartialPhaseIdentification(Feeder):
    """
    Subclass of Feeder which contains functionality to do a partial phase identification (i.e. only identify a subset
    of customers) during one iteration. This is used by the load based methods which solve the phase Identification partly
    and then subtract the load profile from the correct phase.
    """

    def __init__(self, feederID='65019_74469', include_three_phase=False, measurement_error=0.0,length =24):
        """
        Initialize the PartialPhaseIdentification object by reading out the data from JSON files in the specified directory.
        feederID = full identification number of the feeder
        include_three_phase = put on True if you want to include 3 phase customers in your analysis, 3 phase customers
                              will be regarded as 3 single phase customers
        measurement_error = std of the amount of noise added to the voltage (p.u.)
        length = number of data samples used, the first samples are used
        """
        Feeder.__init__(self, feederID, include_three_phase, measurement_error=0, length=length)
        pl = self.phase_labels
        self.add_noise(measurement_error, data="load")
        self.add_noise(measurement_error, data="voltage")
        self.partial_phase_labels = np.array([0] * len(pl))

    def sort_devices_by_variation(self):
        """
        Devices are sorted such that the phases with highest variability are handled first
        using mean absolute variability (MAV)
        """
        lf = self.load_features
        vf = self.voltage_features
        lf_var = self.get_load_variations_matrix()
        i = np.array(self.device_IDs)

        lf_mav = []
        for col in lf_var:
            lf_mav.append(sum(abs(col))/len(col))
        lf_mav = np.array(lf_mav)
        sort_order = lf_mav.argsort()
        self.device_IDs = i[sort_order[::-1]]
        self.load_features = lf[sort_order[::-1]]
        self.voltage_features = vf[sort_order[::-1]]
        self.phase_labels = np.array(self.phase_labels)[sort_order[::-1]]

    def sub_load_profile(self,j,phase):
        """
        Subtracts the load profile from the total and assigns a phase label "phase" to device on index "j"
        """
        self.load_features_transfo[phase - 1] -= self.load_features[j]
        self.partial_phase_labels[j] = phase

    def get_load_variations_matrix(self):
        var = []
        for row in self.load_features:
            new_row = [0] * len(row)
            for i in range(1, len(row)):
                new_row[i] = row[i] - row[i - 1]
            var.append(new_row)
        return np.array(var)

    def get_voltage_variations_matrix(self):
        var = []
        for row in self.voltage_features:
            new_row = [0] * len(row)
            for i in range(1, len(row)):
                new_row[i] = row[i] - row[i - 1]
            var.append(new_row)
        return np.array(var)

    def get_transfo_load_variations_matrix(self):
        var_tot = []
        for row in self.load_features_transfo:
            new_row = [0] * len(row)
            for i in range(1, len(row)):
                new_row[i] = row[i] - row[i - 1]
            var_tot.append(new_row)
        return np.array(var_tot)

    def get_transfo_voltage_variations_matrix(self):
        var_tot = []
        for row in self._voltage_features_transfo:
            new_row = [0] * len(row)
            for i in range(1, len(row)):
                new_row[i] = row[i] - row[i - 1]
            var_tot.append(new_row)
        return np.array(var_tot)

    def get_salient_variations(self, treshold, var, var_transfo):
        """
        Sets the salient variations taking into account the remaining devices and threshold
        """
        sal = []
        sal_transfo = []
        for j in range(0,len(self.phase_labels)):
            new_row = []
            new_row_transfo_a = []
            new_row_transfo_b = []
            new_row_transfo_c = []
            for t in range(1, len(var[0])):
                pl = np.array(self.partial_phase_labels)
                if abs(var[j,t]) > abs(treshold*(sum(var_transfo[:, t])) / len(pl[pl == 0])):
                    new_row += [var[j, t]]
                    new_row_transfo_a += [var_transfo[0, t]]
                    new_row_transfo_b += [var_transfo[1, t]]
                    new_row_transfo_c += [var_transfo[2, t]]
            sal.append(new_row)
            sal_transfo.append([new_row_transfo_a,new_row_transfo_b,new_row_transfo_c])

        sal = np.array(sal)
        sal_transfo = np.array(sal_transfo)
        return sal, sal_transfo


    def find_phase(self, sal, sal_transfo):
        """
        Chooses phase with highest correlation to device j,
        based on it's salient factors sal (and the indexes therof sal_i
        """
        if len(sal) == 0:
            raise AssertionError("No salient components found")
        elif len(sal) < 3:
            best_corr = -np.inf
            for phase in range(0,3):
                corr = sal[0] / sal_transfo[phase][0]
                if corr > best_corr:
                    best_corr = corr
                    best_phase = phase + 1
        else:
            mean_sal = np.mean(sal)
            std_sal = np.std(sal)
            best_phase = 0
            best_corr = -np.inf
            lf = self.load_features_transfo
            for phase in range(0, 3):
                sal_phase = sal_transfo[phase]  # check if this is right
                mean_sal_phase = np.mean(sal_phase)
                std_sal_phase = np.std(sal_phase)
                corr = 1.0/(len(sal)-1) * sum(np.multiply((sal-mean_sal), (sal_phase-mean_sal_phase)) /
                                       np.multiply(std_sal, std_sal_phase))
                if corr >= best_corr:
                    best_corr = corr
                    best_phase = phase + 1
                #print(corr, " ", best_corr, " ", sal[j])
        return best_phase, best_corr

    def assign_devices(self, sal_treshold=0.01, corr_treshold=0.0,sal_components=1):
        """
        """
        var = np.diff(self.load_features(),1)
        var_transfo = np.diff(self.load_features_transfo(),1)
        sal, sal_transfo = self.get_salient_variations(sal_treshold, var, var_transfo)
        counter = 0
        for j in range(0,len(self.device_IDs)):
            if len(sal[j]) > 0 and self.partial_phase_labels[j] == 0:
                phase, corr = self.find_phase(sal[j], sal_transfo[j])
                if corr > corr_treshold:
                    self.sub_load_profile(j, phase)
                    counter += 1
                    var_transfo = self.get_transfo_load_variations_matrix()
                #else:
                    #print(corr, "is below correlation threshold")

        progress = sum(np.array(self.partial_phase_labels) != 0) / len(self.partial_phase_labels)
        acc = self.accuracy()
        print(counter, " devices allocated, ", progress*100, "% done, accuracy ", acc*100, "%")
        return progress

    def load_correlation(self,sal_treshold=0.1, corr_treshold=0.2,sal_components=1):
        """
        Implements load correlation algorithm as described by Li et. al.
        Continues to assign devices as long as progress is being made
        """
        progress = 0.0
        last_progress = 1.0

        self.sort_devices_by_variation()

        while progress < 1.0 and last_progress != progress:
            last_progress = progress
            progress = self.assign_devices(sal_treshold, corr_treshold,sal_components)

    def accuracy(self):
        if len(self.partial_phase_labels) != len(self.phase_labels):
            raise IndexError("Phase labels not of same length")
        c = 0.0
        for i in range(0, len(self.partial_phase_labels)):
            if self.partial_phase_labels[i] == self.phase_labels[i]:
                c = c + 1.0
        try:
            acc = c / (len(self.phase_labels))
        except ZeroDivisionError:
            acc = np.nan
        return acc


    def add_noise(self, error=0, data="voltage"):
        if data == "voltage":
            noise = np.random.normal(0, error/3, [np.size(self.voltage_features, 0), np.size(self.voltage_features, 1)])
            self.voltage_features = self.voltage_features + noise
        if data == "load":
            error = error *0.007/3
            noise = np.random.normal(0, error, [np.size(self.load_features, 0), np.size(self.load_features, 1)])
            self.load_features = self.load_features + noise

    def reset_partial_phase_identification(self):
        self.partial_phase_labels = np.array([0] * len(self.phase_labels))

    def reset_load_features_transfo(self):
        self_copy = copy.deepcopy(self)
        self.load_features_transfo = getattr(self_copy,"_load_features_transfo")


class PartialMissingPhaseIdentification(PartialPhaseIdentification):
    def __init__(self, feederID = '65019_74469', include_three_phase = False, measurement_error = 0.0, length = 24, missing_ratio = 0.0):
        PartialPhaseIdentification.__init__(self, feederID, include_three_phase, measurement_error=measurement_error, length=length)
        self.nb_missing = 0
        for col in self.load_features:
            if sum(col) == 0:
                self.nb_missing += 1
        nb_to_add = round(len(self.phase_labels)*missing_ratio) - self.nb_missing

        while nb_to_add > 0:
            self.load_features[random.randint(0, len(self.phase_labels) - 1)] = np.zeros(self.length)
            nb_to_add -= 1

    def add_missing(self,ratio):
        nb = round(len(self.phase_labels)*ratio)
        raise NotImplementedError

    def accuracy(self):
        if len(self.partial_phase_labels) != len(self.phase_labels):
            raise IndexError("Phase labels not of same length")
        c = 0.0
        for i in range(0, len(self.partial_phase_labels)):
            if self.partial_phase_labels[i] == self.phase_labels[i]:
                c = c + 1.0
        try:
            acc = c / (len(self.phase_labels)-len(self.missing))
        except ZeroDivisionError:
            acc = np.nan
        return acc
