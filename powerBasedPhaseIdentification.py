from common import *


class PartialPhaseIdentification(Feeder):
    """
    Subclass of Feeder which contains
    """

    def __init__(self, feederID='65019_74469', include_three_phase=False, measurement_error=0.0,):
        Feeder.__init__(self, feederID, include_three_phase, measurement_error=0.0)
        pl = self.phase_labels
        lf = self.load_features
        self._load_features_total = np.zeros([3, len(lf[1])])
        for i, col in enumerate(lf):
            self._load_features_total[pl[i]-1] += col
        self.add_noise(measurement_error, data="load")
        self.partial_phase_labels = [0] * len(pl)

    def sort_devices_by_variation(self):
        """
        Devices are sorted such that the phases with highest variability are handled first
        using mean absolute variability (MAV)
        """
        lf = self.load_features
        lf_var = self.get_variations_matrix()
        i = np.array(self.device_IDs)

        lf_mav = []
        for col in lf_var:
            lf_mav.append(sum(abs(col))/len(col))
        lf_mav = np.array(lf_mav)
        sort_order = lf_mav.argsort()
        self.device_IDs = i[sort_order[::-1]]
        self.load_features = lf[sort_order[::-1]]
        self.phase_labels = np.array(self.phase_labels)[sort_order[::-1]]

    def sub_load_profile(self,j,phase):
        """
        Subtracts the load profile from the total and assigns a phase label
        """
        self._load_features_total[phase-1] -= self.load_features[j]
        self.partial_phase_labels[j] = phase


    def get_variations_matrix(self):
        var = []
        for row in self.load_features:
            new_row = [0] * len(row)
            for i in range(1, len(row)):
                new_row[i] = row[i] - row[i - 1]
            var.append(new_row)
        return np.array(var)

    def get_total_variations_matrix(self):
        var_tot = []
        for row in self._load_features_total:
            new_row = [0] * len(row)
            for i in range(1, len(row)):
                new_row[i] = row[i] - row[i - 1]
            var_tot.append(new_row)
        return np.array(var_tot)

    def get_salient_variations(self,treshold):
        """
        Sets the salient variations taking into account the remaining devices and threshold
        """
        var = self.get_variations_matrix()
        var_tot = self.get_total_variations_matrix()

        sal = []
        sal_i = []
        for j in range(0,len(self.phase_labels)):
            new_row = []
            new_row_i = []
            for t in range(1, len(var[0])):
                pl = np.array(self.partial_phase_labels)
                if abs(var[j,t]) > treshold*(sum(var_tot[:, t])-var[j, t]) / len(pl[pl == 0]):
                    new_row += [var[j, t]]
                    new_row_i += [t]
            sal.append(new_row)
            sal_i.append(new_row_i)
        sal = np.array(sal)
        sal_i = np.array(sal_i)
        return sal, sal_i, var_tot, var

    def find_phase(self, sal, sal_i, var_tot):
        """
        Chooses phase with highest correlation to device j,
        based on it's salient factors sal (and the indexes therof sal_i
        """
        if len(sal) == 0:
            raise AssertionError("No salient components found for ",j)
        elif len(sal) < 3:
            best_corr = -np.inf
            for phase in range(0,3):
                corr = sal[0] / var_tot[phase][sal_i][0]
                if corr > best_corr:
                    best_corr = corr
                    best_phase = phase + 1
        else:
            mean_sal = np.mean(sal)
            std_sal = np.std(sal)
            best_phase = 0
            best_corr = -np.inf
            lf = self._load_features_total
            for phase in range(0,3):
                sal_phase = var_tot[phase][sal_i] # check if this is right
                mean_sal_phase = np.mean(sal_phase)
                std_sal_phase = np.std(sal_phase)
                corr = 1.0/(len(sal)-1) * sum(np.multiply((sal-mean_sal), (sal_phase-mean_sal_phase)) /
                                       np.multiply(std_sal, std_sal_phase))
                if corr >= best_corr:
                    best_corr = corr
                    best_phase = phase + 1
                #print(corr, " ", best_corr, " ", sal[j])
        return best_phase, best_corr

    def find_easy_device(self, sal_treshold=0.01, corr_treshold=0.0):
        """
        """
        sal, sal_i, var_tot, var = self.get_salient_variations(treshold=sal_treshold)
        counter = 0
        for j in range(0,len(self.device_IDs)):
            if len(sal[j]) > 0 and self.partial_phase_labels[j] == 0:
                phase, corr = self.find_phase(sal[j],sal_i[j],var_tot)
                if corr > corr_treshold:
                    self.sub_load_profile(j,phase)
                    counter += 1
                    var_tot = self.get_total_variations_matrix()

        progress = sum(np.array(self.partial_phase_labels) != 0) / len(self.partial_phase_labels)
        acc = self.accuracy()
        print(counter, " devices allocated, ", progress*100, "% done, accuracy ", acc*100, "%")
        return progress

    def load_correlation(self,sal_treshold=0.1, corr_treshold=0.2):
        progress = 0.0
        last_progress = 1.0

        self.sort_devices_by_variation()

        while progress < 1.0 and last_progress != progress:
            last_progress = progress
            progress = self.find_easy_device(sal_treshold, corr_treshold)

    def accuracy(self):
        if len(self.partial_phase_labels) != len(self.phase_labels):
            raise IndexError("Phase labels not of same length")
        c = 0.0
        for i in range(0, len(self.partial_phase_labels)):
            if self.partial_phase_labels[i] == self.phase_labels[i]:
                c = c + 1.0
        return c / len(self.partial_phase_labels)

    def add_noise(self, error=0, data="voltage"):
        if data == "voltage":
            noise = np.random.normal(0, error, [np.size(self.voltage_features, 0), np.size(self.voltage_features, 1)])
            self.voltage_features = self.voltage_features + noise
        if data == "load":
            error = error * np.mean(self.load_features)
            noise = np.random.normal(0, error, [np.size(self.load_features, 0), np.size(self.load_features, 1)])
            self.load_features = self.load_features + noise
