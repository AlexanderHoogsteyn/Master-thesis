from common import *

class PhaseIdentification(Feeder):
    """
    A phaseIdentification object is formed by performing one of the phase identification methods on the feeder object
    It contains most notably an array with the found phase labels by the method
    """

    def __init__(self, feederID='65019_74469', include_three_phase=False, measurement_error=0.0):
        Feeder.__init__(self, feederID, include_three_phase, measurement_error)

        self._score = np.nan

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
        self._algorithm = 'hierarchal clustering'
        self._n_repeats = 1
        self.partial_phase_labels = labels + 1
        self.match_labels()

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

        self._algorithm = 'k-means++'
        self._n_repeats = n_repeats
        self.partial_phase_labels = best_cluster_labels + 1
        self.match_labels()

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
        self._algorithm = 'k-medoids++'
        self._n_repeats = n_repeats
        self.partial_phase_labels = best_cluster_labels + 1
        self.match_labels()

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

        self._algorithm = 'Gaussian mixture model'
        self._n_repeats = n_repeats
        self.partial_phase_labels = best_cluster_labels + 1
        self.match_labels()
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
            labels = self.phase_labels
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
        self._algorithm = 'voltage_correlation'
        self._n_repeats = 1
        self.partial_phase_labels = phase_labels

    def accuracy(self):
        correct_labels = self.partial_phase_labels
        labels = self.phase_labels
        if len(labels) != len(correct_labels):
            raise IndexError("Phase labels not of same length")
        c = 0.0
        for i in range(0, len(labels)):
            if labels[i] == correct_labels[i]:
                c = c + 1.0
        return c / len(labels)

    def find_wrong_IDs(self):
        correct_labels = self.phase_labels
        labels = self.partial_phase_labels
        id_s = self.device_IDs
        wrong_ids = []
        if len(labels) != len(correct_labels):
            raise IndexError("Phase labels not of same length")
        for i in range(0, len(labels)):
            if labels[i] != correct_labels[i]:
                wrong_ids += [id_s[i]]
        return np.array(wrong_ids)

    def match_labels(self):
        best_labels = self.partial_phase_labels
        best_acc = 0.0
        for i in range(0, 7):
            acc = self.accuracy()
            labels = self.partial_phase_labels
            if acc > best_acc:
                best_acc = acc
                best_labels = labels
            if i == 3:
                for j in range(0, len(labels)):
                    if labels[j] == 1:
                        labels[j] = 2
                    elif labels[j] == 2:
                        labels[j] = 1
                self.partial_phase_labels = np.array(labels)
            else:
                self.partial_phase_labels = (list(map(lambda x: x % 3 + 1, labels)))
        self.partial_phase_labels = np.array(best_labels)

    def plot_voltages(self, length=48, x_axis=None, y_axis=None):
        """
        Makes a 2D plot of the resulting clusters. You need to specify the Feeder object which contains all the used data
        as well as the Cluster object which you obtained by performing one on the clustering algorithm methods
        on the Feeder.
        It can be chosen what is plotted on the x and y axis by specifying the name of a feature. This has to be the specific
        string corresponding to that feature such as "Yearly consumption per customer (kWh)" (These can be found using
        Feeder.get_feature_list() ).
        """
        voltage_data = self.voltage_features
        plt.figure(figsize=(8, 6))
        markers = ["s", "o", "D", ">", "<", "v", "+"]
        x = np.arange(0, length)
        for i in range(0, 3):
            color = plt.cm.viridis(float(i) / (float(self.nb_customers) - 1.0))
            for line in voltage_data:
                plt.plot(x, line[0:length], color=color, alpha=0.85)
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.title(self._algorithm)
        plt.show()

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
