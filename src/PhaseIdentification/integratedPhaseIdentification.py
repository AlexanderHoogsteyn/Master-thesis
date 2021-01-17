from src.PhaseIdentification.powerBasedPhaseIdentification import *
from src.VisualizePhaseIdentification.visualization import *


class IntegratedPhaseIdentification(PartialPhaseIdentification):

    def __init__(self, feederID='65019_74469', include_three_phase=False, measurement_error=0.0, length=24):
        PartialPhaseIdentification.__init__(self, feederID, include_three_phase, measurement_error, length=length)

    def voltage_assisted_load_correlation(self, sal_treshold_load=1, sal_treshold_volt=0.4, corr_treshold=0.2,
                                          volt_assist=0.0, length=24):
        """
        Also uses salient voltage measurements, therefore also feeder voltage needed
        """
        counter = 1
        completeness = 0

        # Sorting done according to highest variance in load
        self.sort_devices_by_variation()
        C = CorrelationCoeficients(self)
        C.visualize_correlation_all()

        while counter > 0 and completeness != 1:

            # Load salient components
            var_load = np.diff(self.load_features[:, 0:length], 1)
            var_transfo_load = np.diff(self.load_features_transfo[:, 0:length], 1)
            sal_load, sal_transfo_load = self.get_salient_variations(sal_treshold_load, var_load, var_transfo_load)
            # print("# Salient components load between ", min(nb_sal), " and ", max(nb_sal))

            # Voltage salient components
            var_volt = np.diff(self.voltage_features[:, 0:length], 1)
            var_transfo_volt = np.diff(self._voltage_features_transfo[:, 0:length], 1)
            # sal_volt, sal_transfo_volt = self.get_salient_variations(sal_treshold_volt, var_volt, var_transfo_volt)
            # ("# Salient components voltage between ", min(nb_sal), " and ", max(nb_sal))

            # Reactive power salient components?

            counter = 0
            for j in range(0, len(self.device_IDs)):
                if self.partial_phase_labels[j] == 0:
                    phase, corr = self.find_phase(var_volt[j], var_transfo_volt, sal_load[j], sal_transfo_load[j],
                                                  volt_assist)
                    if corr > corr_treshold:
                        # Subtract assigned load from transfo measurement & update variance "var_transfo_load"
                        self.sub_load_profile(j, phase)
                        var_transfo_load = self.get_transfo_load_variations_matrix()
                        counter += 1

                    # else:
                    #   print(corr, "is below correlation threshold")



            completeness = sum(np.array(self.partial_phase_labels) != 0) / len(self.partial_phase_labels)
            acc = self.accuracy()
            print(counter, " devices allocated, ", completeness * 100, "% done, accuracy ", acc * 100, "%")
            C = CorrelationCoeficients(self)
            C.visualize_correlation_all()
        if completeness != 1:
            # Complete remaining
            # Load salient components
            var_load = np.diff(self.load_features[:, 0:length], 1)
            var_transfo_load = np.diff(self.load_features_transfo[:, 0:length], 1)
            sal_load, sal_transfo_load = self.get_salient_variations(sal_treshold_load, var_load, var_transfo_load)
            # print("# Salient components load between ", min(nb_sal), " and ", max(nb_sal))

            # Voltage salient components
            var_volt = np.diff(self.voltage_features[:, 0:length], 1)
            var_transfo_volt = np.diff(self._voltage_features_transfo[:, 0:length], 1)
            # sal_volt, sal_transfo_volt = self.get_salient_variations(sal_treshold_volt, var_volt, var_transfo_volt)
            # ("# Salient components voltage between ", min(nb_sal), " and ", max(nb_sal))

            # Reactive power salient components?

            counter = 0
            for j in range(0, len(self.device_IDs)):
                if self.partial_phase_labels[j] == 0:
                    phase, corr = self.find_phase(var_volt[j], var_transfo_volt, sal_load[j], sal_transfo_load[j],
                                                  volt_assist)
                    # Subtract assigned load from transfo measurement & update variance "var_transfo_load"
                    self.sub_load_profile(j, phase)
                    var_transfo_load = self.get_transfo_load_variations_matrix()
                    counter += 1

                    # else:
                    #   print(corr, "is below correlation threshold")

            completeness = sum(np.array(self.partial_phase_labels) != 0) / len(self.partial_phase_labels)
            acc = self.accuracy()
            print(counter, " devices allocated, ", completeness * 100, "% done, accuracy ", acc * 100, "%")
            C = CorrelationCoeficients(self)
            C.visualize_correlation_all()

    def find_phase(self, sal_volt, sal_transfo_volt, sal_load, sal_transfo_load, volt_assist=0):
        """
        Chooses phase with highest correlation to device j,
        based on it's salient factors sal (and the indexes therof sal_i
        """
        load_invalid = False
        voltage_invalid = False

        if len(sal_volt) + len(sal_load) == 0:
            raise AssertionError("No salient components found")
        elif len(sal_volt) < 3 and len(sal_load) < 3:
            best_corr = -np.inf
            for phase in range(0, 3):
                corr = sal_load[0] / sal_transfo_load[phase][0]
                if corr > best_corr:
                    best_corr = corr
                    best_phase = phase + 1
        else:
            mean_sal_volt = np.mean(sal_volt)
            mean_sal_load = np.mean(sal_load)
            std_sal_volt = np.std(sal_volt)
            std_sal_load = np.std(sal_load)
            if std_sal_load == 0:
                load_invalid = True
            if std_sal_volt == 0:
                voltage_invalid = True

            best_phase = 0
            best_corr = -np.inf

            lf = self.load_features_transfo
            vf = self._voltage_features_transfo

            for phase in range(0, 3):
                sal_phase_volt = sal_transfo_volt[phase]
                mean_sal_phase_volt = np.mean(sal_phase_volt)
                std_sal_phase_volt = np.std(sal_phase_volt)
                if std_sal_phase_volt == 0 or len(sal_volt) == 0:
                    volt_invalid = True

                try:
                    corr_volt = 1.0 / (len(sal_volt) - 1) * sum(
                        (sal_volt - mean_sal_volt) * (sal_phase_volt - mean_sal_phase_volt)) \
                                / (std_sal_volt * std_sal_phase_volt)
                except ZeroDivisionError:
                    corr_volt = np.nan
                    voltage_invalid = True

                sal_phase_load = sal_transfo_load[phase]
                mean_sal_phase_load = np.mean(sal_phase_load)
                std_sal_phase_load = np.std(sal_phase_volt)
                if std_sal_phase_load == 0 or len(sal_load) == 0:
                    load_invalid = True

                try:
                    corr_load = 1.0 / (len(sal_load) - 1) * sum(
                        (sal_load - mean_sal_load) * (sal_phase_load - mean_sal_phase_load)) \
                                / (std_sal_load * std_sal_phase_load)
                except ZeroDivisionError:
                    corr_load = np.nan
                    load_invalid = True

                if corr_load == np.nan:
                    corr_load = -np.inf
                # print(phase, corr_volt)
                # Combine correlations
                if load_invalid:
                    corr = corr_volt
                elif voltage_invalid:
                    corr = corr_load
                else:
                    corr = (1 - volt_assist) * corr_load + volt_assist * corr_volt
                if corr > best_corr:
                    best_corr = corr
                    best_phase = phase + 1
                # print(corr, " ", best_corr, " ", sal[j])
        # print("Phase label: ", best_phase, "load corr ", corr_load, load_invalid, "voltage corr ", corr_volt, voltage_invalid)
        return best_phase, best_corr


class IntegratedMissingPhaseIdentification(IntegratedPhaseIdentification):
    def __init__(self, feederID='65019_74469', include_three_phase=False, measurement_error=0.0, length=24,
                 missing_ratio=0.0):
        IntegratedPhaseIdentification.__init__(self, feederID, include_three_phase, measurement_error=measurement_error,
                                               length=length)
        self.nb_missing = 0
        self.nb_original_devices = len(self.phase_labels)
        nb_to_add = round(len(self.phase_labels) * missing_ratio) - self.nb_missing
        if nb_to_add < 0:
            nb_to_add = 0

        i_to_remove = np.random.choice(np.arange(len(self.phase_labels)), nb_to_add, replace=False)
        self.load_features = np.delete(self.load_features, i_to_remove, axis=0)
        self.voltage_features = np.delete(self.voltage_features, i_to_remove, axis=0)
        self.phase_labels = np.delete(self.phase_labels, i_to_remove, axis=0)
        self.partial_phase_labels = np.delete(self.partial_phase_labels, i_to_remove, axis=0)

        self.nb_missing += len(i_to_remove)
        self.missing_ratio = missing_ratio

    def add_missing(self, ratio):
        nb = round(self.nb_original_devices * ratio)
        nb_to_add = nb - self.nb_missing
        if nb_to_add < 0:
            nb_to_add = 0
        i_to_remove = np.random.choice(np.arange(len(self.phase_labels)), nb_to_add, replace=False)
        self.load_features = np.delete(self.load_features, i_to_remove, axis=0)
        self.voltage_features = np.delete(self.voltage_features, i_to_remove, axis=0)
        self.phase_labels = np.delete(self.phase_labels, i_to_remove, axis=0)
        self.partial_phase_labels = np.delete(self.partial_phase_labels, i_to_remove, axis=0)

        self.nb_missing += len(i_to_remove)
        self.missing_ratio = ratio
