from phase_identification import *

"""
##################################################
#   CASE A: Rural area  
#           feeder ID = 86315_785383
#           number of devices = 22
#   CASE B: Urban area
#           number of devices = 125
#           feeder ID = 65028_84566
#   CASE C: Average feeder
#           number of devices = 76
#           feeder ID = 1830188_2181475
##################################################
"""
include_A = True
include_B = True
include_C = True
voltage_noise = 0.00
load_noise = 0.03   #pu
include_three_phase = False
length = 24*7
n_repeats = 1

"""
Choose data representation between: "raw", "delta", "binary" or
do a comparison using "comparison"
Fluvinus experiment can be done using "truncated"
"""
representation = "raw"

"""
Choose Algorithm between: "clustering", "correlation", "load-correlation"
"""
algorithm = "load_correlation"

included_feeders = []
if include_A:
    included_feeders.append("86315_785383")
if include_B:
    included_feeders.append("65028_84566")
if include_C:
    included_feeders.append("1830188_2181475")

for feeder_id in included_feeders:
    feeder = Feeder(measurement_error=voltage_noise, feederID=feeder_id, include_three_phase=include_three_phase)
    feeder_copy = copy.deepcopy(feeder)
    if representation == "comparison":
        error_range = np.arange(0, 0.02, 0.0005)
        results, scores = compare_data_representations(feeder_copy, error_range, n_repeats=n_repeats)
        for i in range(0, 9):
            results, scores_sum = compare_data_representations(feeder_copy, error_range, n_repeats=n_repeats)
            scores = scores + scores_sum
        scores = scores/10
        plt.plot(error_range * 100, scores[0, :] * 100, label="Raw data")
        plt.plot(error_range * 100, scores[1, :] * 100, label="Delta data")
        plt.plot(error_range * 100, scores[2, :] * 100, label="Binary data")
        plt.ylabel("Accuracy (%)")
        plt.xlabel("Standard deviation of measurement error (%) ")
        plt.legend()
        plt.show()
    elif representation == "delta":
        feeder.change_data_representation(representation="delta")
    elif representation == "binary":
        feeder.change_data_representation(representation="binary")
    elif representation == "truncated":
        feeder.truncate_voltages()

    if algorithm == "clustering":
        clusters = feeder.k_means_clustering(3, n_repeats=n_repeats)
        clusters.match_labels(feeder)
        print("Accuracy using ", representation, " data: ", clusters.accuracy(feeder))
        feeder.plot_voltages(length=length)
        feeder.plot_load_profiles(length=length)
        #silhouette_analysis(feeder, clusters)

    if algorithm == "clustering_comparison":
        compare_algorithms(feeder, 'accuracy', n_repeats, range=range(2, 5))

    elif algorithm == "correlation":
        identification = feeder.voltage_correlation()
        print("Accuracy using voltage correlation: ", identification.accuracy(feeder))
        print("wrong device ID's: ", identification.find_wrong_IDs(feeder))

    elif algorithm == "load_correlation":
        load_feeder = PartialPhaseIdentification(measurement_error=load_noise, feederID=feeder_id, include_three_phase=include_three_phase)
        print("Start load correlation algorithm for ", feeder_id)
        load_feeder.load_correlation(sal_treshold=0.2, corr_treshold=0.0)