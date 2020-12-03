

"""
Toolset for identifying phase connectivity of customers in a distribution network.
Author: Alexander Hoogsteyn
Date:   16-09-2020
"""


class Feeder(object):
    """
    A Feeder object contains all the data (voltage + load) that is needed to perform the clustering. The path_topology attribute is
    used to specify the folder which contains the JSON files. The object will store the features in  a numpy array as well as some metadata
    such as list of the features used and the ID's of the feeders.
    """

    def __init__(self, path_data='C:/Users/AlexH/OneDrive/Documenten/Julia/Master-thesis/POLA_data/',
                 path_topology='C:/Users/AlexH/OneDrive/Documenten/Julia/Master-thesis/POLA/',
                 feederID='65019_74469', include_three_phase=False, measurement_error=0.0):
        """
        Initialize the feeder object by reading out the data from JSON files in the specified directory
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
                    self.multiphase_IDs += [deviceID]
                    self.device_IDs += [deviceID, deviceID, deviceID]
                else:
                    self.device_IDs += [deviceID]
                self.phase_labels += device_phases
        noise = np.random.normal(0, measurement_error, [np.size(voltage_features, 0), np.size(voltage_features, 1)])
        self.voltage_features = np.array(voltage_features) + noise
        self.load_features = np.array(load_features)
        self.phase_labels = np.array(self.phase_labels)


    def get_IDs(self):
        """
        Method to obtain a numpy array of the feeders used, the indeces will correspond to the indeces on the rows
        obtained using get_features(), get_feature() or Clusters.get_clusters()
        """
        return self.device_IDs

    def get_phase_labels(self):
        """
        Method to obtain a list of the features used, the order of which will correspond to the order of the columns in
        get_features()
        """
        return self.phase_labels

    def get_nb_customers(self):
        return self.nb_customers

    def hierarchal_clustering(self, n_clusters=3, normalized=True, criterion='avg_silhouette'):
        """
        Method that returns a clustering object obtained by performing hierarchal clustering of the specified featureset
        By default the features will be normalized first. By scaling the features to have a mean of 0 and unit variance.
        (More info: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
        """
        if normalized:
            scaler = StandardScaler()
            data = scaler.fit_transform(self.voltage_features)
        else:
            data = self.voltage_features
        labels = AgglomerativeClustering(n_clusters).fit(data).labels_
        if criterion == 'global_silhouette':
            score = global_silhouette_criterion(data, labels)
        if criterion == 'avg_silhouette':
            score = silhouette_score(data, labels)
        return Cluster(labels, 'hierarchal clustering', normalized, 1, criterion, score)

    def k_means_clustering(self, n_clusters=3, normalized=True, n_repeats=1, criterion='avg_silhouette'):
        """
        Method that returns a clustering object obtained by performing K-means++ on the specified feeder.
        A number of repetitions can be specified, the best result according to the specified criterion will be returned
        By default the features will be normalized first. By scaling the features to have a mean of 0 and unit variance.
        (More info: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
        """
        if normalized == True:
            scaler = StandardScaler()
            data = scaler.fit_transform(self.voltage_features)
        else:
            data = self.voltage_features

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
        Method that returns a clustering object obtained by performing K-medoids++ on the specified feeder.
        A number of repetitions can be specified, the best result according to the specified criterion will be returned
        By default the features will be normalized first. By scaling the features to have a mean of 0 and unit variance.
        (More info: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
        """
        if normalized == True:
            scaler = StandardScaler()
            data = scaler.fit_transform(self.voltage_features)
        else:
            data = self.voltage_features

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
        Method that returns a clustering object obtained by performing K-means++ on the specified feeder.
        A number of repetitions can be specified, the best result according to the average silhouette score will be
        returned
        By default the features will be normalized first. By scaling the features to have a mean of 0 and unit variance.
        (More info: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
        """
        if normalized == True:
            scaler = StandardScaler()
            data = scaler.fit_transform(self.voltage_features)
        else:
            data = self.voltage_features

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
        id_ = np.array(self.device_IDs)
        try:
            id_3 = self.multiphase_IDs[0]
        except IndexError:
            raise ValueError("No 3 phase reference found")
        else:
            profiles = self.voltage_features
            profiles = profiles[id_ == id_3]
            labels = self.get_phase_labels()
            labels = labels[id_ == id_3]
        return labels, profiles

    def voltage_correlation(self):
        labels, profiles = self.get_reference_3phase_customer()
        phase_labels = []
        scores = []
        for device in self.voltage_features:
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
        return self.plot_data(self.voltage_features, ylabel, length)

    def plot_load_profiles(self, ylabel="Power (kW)", length=48):
        return self.plot_data(self.load_features * 500, ylabel, length)

    def change_data_representation(self, representation="delta", data="voltage", inplace=True):
        if data == "voltage":
            original_data = self.voltage_features
        elif data == "load":
            original_data = self.load_features
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
                self.voltage_features = np.array(new_data)
            if data == "load":
                self.load_features = np.array(new_data)
        else:
            new_self = copy.deepcopy(self)
            if data == "voltage":
                new_self.voltage_features = np.array(new_data)
            if data == "load":
                new_self.load_features = np.array(new_data)
            return new_self

    def truncate_voltages(self):
        vf = self.voltage_features
        vf = vf*230     #chanve from pu to V
        vf = np.trunc(vf)
        self.voltage_features = vf


    def add_noise(self, error=0, data="voltage",inplace=True):
        if inplace:
            if data == "voltage":
                voltage_features = self.voltage_features
                noise = np.random.normal(0, error, [np.size(voltage_features, 0), np.size(voltage_features, 1)])
                self.voltage_features = voltage_features + noise
            if data == "load":
                error = error * np.mean(self.load_features)
                noise = np.random.normal(0, error, [np.size(self.load_features, 0), np.size(load_features, 1)])
                self.load_features = self.load_features + noise
        else:
            if data == "voltage":
                voltage_features = self.voltage_features
                noise = np.random.normal(0, error, [np.size(voltage_features, 0), np.size(voltage_features, 1)])
                new_self = copy.deepcopy(self)
                new_self.voltage_features = voltage_features + noise
            if data == "load":
                error = error * np.mean(self.load_features)
                noise = np.random.normal(0, error, [np.size(self.load_features, 0), np.size(load_features, 1)])
                new_self = copy.deepcopy(self)
                new_self.load_features = self.load_features + noise
            return new_self



class PartialPhaseIdentification(Feeder):

    def __init__(self, path_data='C:/Users/AlexH/OneDrive/Documenten/Julia/Master-thesis/POLA_data/',
                 path_topology='C:/Users/AlexH/OneDrive/Documenten/Julia/Master-thesis/POLA/',
                 feederID='65019_74469', include_three_phase=False, measurement_error=0.0,):
        Feeder.__init__(self, path_data, path_topology, feederID, include_three_phase, measurement_error=0.0)
        pl = self.get_phase_labels()
        lf = self.load_features
        self._load_features_total = np.zeros([3, len(lf[1])])
        for i, col in enumerate(lf):
            self._load_features_total[pl[i]-1] += col
        self.add_noise(measurement_error, data="load")
        self._partial_phase_labels = [0]*len(pl)

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
        self._device_IDs = i[sort_order[::-1]]
        self.set_load_features(lf[sort_order[::-1]])
        self._phase_labels = np.array(self.get_phase_labels())[sort_order[::-1]]

    def sub_load_profile(self,j,phase):
        """
        Subtracts the load profile from the total and assigns a phase label
        """
        self._load_features_total[phase-1] -= self.load_features[j]
        self._partial_phase_labels[j] = phase


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
        for j in range(0,len(self.get_phase_labels())):
            new_row = []
            new_row_i = []
            for t in range(1, len(var[0])):
                pl = np.array(self._partial_phase_labels)
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
            if len(sal[j]) > 0 and self._partial_phase_labels[j] == 0:
                phase, corr = self.find_phase(sal[j],sal_i[j],var_tot)
                if corr > corr_treshold:
                    self.sub_load_profile(j,phase)
                    counter += 1
                    var_tot = self.get_total_variations_matrix()

        progress = sum(np.array(self._partial_phase_labels) != 0) / len(self._partial_phase_labels)
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
        correct_labels = self.get_phase_labels()
        labels = self._partial_phase_labels
        if len(labels) != len(correct_labels):
            raise IndexError("Phase labels not of same length")
        c = 0.0
        for i in range(0, len(labels)):
            if labels[i] == correct_labels[i]:
                c = c + 1.0
        return c / len(labels)

def silhouette_analysis(Feeder, Cluster):
    """
    Makes a silhouette analysis of the resulting clusters (more info: https://en.wikipedia.org/wiki/Silhouette_(clustering) ).
    You need to specify the Feeder object which contains all the used data as well as the Cluster object
    which you obtained by performing one on the clustering algorithm methods on the Feeder.
    """
    features = Feeder.voltage_features
    cluster_labels = Cluster.get_phase_labels()
    n_clusters = Cluster.get_n_clusters()
    plt.figure(figsize=(22, 10))
    plt.xlim([-0.1, 1])
    plt.ylim([0, len(features) + (n_clusters + 1) * 10])

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
    for i in range(1, n_clusters+1):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = plt.cm.viridis(float(i) / (n_clusters))

        plt.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    plt.title("The silhouette plot for the various clusters.")
    plt.xlabel("The silhouette coefficient values")
    plt.ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    plt.axvline(x=silhouette_avg, color="grey", linestyle="--", label='average silhouette coef %f3' % silhouette_avg)
    plt.axvline(x=silhouette_global, color="grey", linestyle="-.",
                label='global silhouette coef %f3' % silhouette_global)
    plt.legend(loc='upper right')

    plt.yticks([])  # Clear the yaxis labels / ticks
    plt.xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
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

