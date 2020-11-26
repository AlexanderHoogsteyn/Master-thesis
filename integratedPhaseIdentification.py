from powerBasedPhaseIdentification import *


class IntegratedPhaseIdentification(PartialPhaseIdentification):

    def __init__(self, feederID='65019_74469', include_three_phase=False, measurement_error=0.0,length=24):
        PartialPhaseIdentification.__init__(self, feederID, include_three_phase, measurement_error, length=length)

    def voltage_assisted_load_correlation(self, sal_treshold_load=0.4, sal_treshold_volt=0.4, corr_treshold=0.2, volt_assist=0.0):
        """
        Also uses salient voltage measurements, therefore also feeder voltage needed
        """
        progress = 0.0
        last_progress = 1.0

        # Sorting done according to highest variance in load
        self.sort_devices_by_variation()

        while progress < 1.0 and last_progress != progress:
            last_progress = progress

            # Load salient components
            var_load = self.get_load_variations_matrix()
            var_transfo_load = self.get_transfo_load_variations_matrix()
            sal_load, sal_transfo_load = self.get_salient_variations(sal_treshold_load, var_load, var_transfo_load)
            nb_sal = [len(i) for i in sal_load]
            #print("# Salient components load between ", min(nb_sal), " and ", max(nb_sal))

            # Voltage salient components
            var_volt = self.get_voltage_variations_matrix()
            var_transfo_volt = self.get_transfo_voltage_variations_matrix()
            sal_volt, sal_transfo_volt = self.get_salient_variations(sal_treshold_volt, var_volt, var_transfo_volt)
            nb_sal = [len(i) for i in sal_volt]
            #("# Salient components voltage between ", min(nb_sal), " and ", max(nb_sal))

            # Reactive power salient components?

            counter = 0
            for j in range(0, len(self.device_IDs)):
                if len(sal_load[j]) > 0 and self.partial_phase_labels[j] == 0:
                    phase, corr = self.find_phase(var_volt[j], var_transfo_volt, sal_load[j], sal_transfo_load[j], volt_assist)
                    if corr > corr_treshold:
                        # Subtract assigned load from transfo measurement & update variance "var_transfo_load"
                        self.sub_load_profile(j, phase)
                        var_transfo_load = self.get_transfo_load_variations_matrix()
                        counter += 1

                    #else:
                     #   print(corr, "is below correlation threshold")

            progress = sum(np.array(self.partial_phase_labels) != 0) / len(self.partial_phase_labels)
            acc = self.accuracy()
            print(counter, " devices allocated, ", progress * 100, "% done, accuracy ", acc * 100, "%")

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
            for phase in range(0,3):
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
                if std_sal_phase_volt == 0:
                    volt_invalid = True

                corr_volt = 1.0/(len(sal_volt)-1) * sum(np.multiply((sal_volt-mean_sal_volt), (sal_phase_volt-mean_sal_phase_volt)) /
                                       np.multiply(std_sal_volt, std_sal_phase_volt))

                sal_phase_load = sal_transfo_load[phase]
                mean_sal_phase_load = np.mean(sal_phase_load)
                std_sal_phase_load = np.std(sal_phase_volt)
                if std_sal_phase_load == 0:
                    load_invalid = True

                corr_load = 1.0/(len(sal_load)-1) * sum(np.multiply((sal_load-mean_sal_load), (sal_phase_load-mean_sal_phase_load)) /
                                       np.multiply(std_sal_load, std_sal_phase_load))
                if corr_load == np.nan:
                    corr_load = -np.inf
                #print(phase, corr_volt)
                # Combine correlations
                if load_invalid:
                    corr = corr_volt
                elif voltage_invalid:
                    corr = corr_load
                else:
                    corr = (1-volt_assist)*corr_load + volt_assist*corr_volt
                print(corr)
                if corr > best_corr:
                    best_corr = corr
                    best_phase = phase + 1
                #print(corr, " ", best_corr, " ", sal[j])
        return best_phase, best_corr


class IntegratedMissingPhaseIdentification(IntegratedPhaseIdentification):
    def __init__(self, feederID = '65019_74469', include_three_phase = False, measurement_error = 0.0, length = 24, missing_ratio = 0.0):
        IntegratedPhaseIdentification.__init__(self, feederID, include_three_phase, measurement_error=measurement_error, length=length)
        self.nb_missing = 0
        for col in self.load_features:
            if sum(col) == 0:
                self.nb_missing += 1
        nb_to_add = round(len(self.phase_labels)*missing_ratio) - self.nb_missing

        while nb_to_add > 0:
            self.load_features[random.randint(0, len(self.phase_labels) - 1)] = np.zeros(self.length)
            nb_to_add -= 1

    def add_missing(self, ratio):
        nb = round(len(self.phase_labels)*ratio)
        raise NotImplementedError