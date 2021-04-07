import sys
from os.path import dirname
sys.path.append(dirname("../src/"))

from src.PhaseIdentification.voltageBasedPhaseIdentification import *

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
voltage_noise = 0.01
include_three_phase = True
length = 24
n_repeats = 1

"""
Choose data representation between: "raw", "delta", "binary" or
do a comparison using "comparison"
"""
representation = "raw"

"""
Choose Algorithm between: "clustering", "correlation", "load-correlation"
"""
algorithm = "correlation"

included_feeders = []
if include_A:
    included_feeders.append("1351982_1596442")
if include_B:
    included_feeders.append("65028_84566")
if include_C:
    included_feeders.append("1830188_2181475")

for feeder_id in included_feeders:
    feeder = Feeder(feederID=feeder_id, include_three_phase=include_three_phase)
    phaseID = PhaseIdentification(feeder, ErrorClass(voltage_noise))

    if representation == "comparison":
        error_range = np.arange(0, 0.02, 0.0005)
        results, scores = compare_data_representations(phaseID, error_range, n_repeats=n_repeats)
        for i in range(0, 9):
            results, scores_sum = compare_data_representations(phaseID, error_range, n_repeats=n_repeats)
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
        phaseID.change_data_representation(representation="delta")
    elif representation == "binary":
        phaseID.change_data_representation(representation="binary")
    elif representation == "truncated":
        phaseID.truncate_voltages()

    if algorithm == "clustering":
        phaseID.k_means_clustering(3, n_repeats=n_repeats)
        print("Accuracy using ", representation, " data: ", phaseID.accuracy())
        phaseID.plot_voltages(length=length)
        phaseID.plot_load_profiles(length=length)
        #silhouette_analysis(feeder, clusters)

    if algorithm == "clustering_comparison":
        compare_algorithms(phaseID, 'accuracy', n_repeats, range=range(2, 5))

    elif algorithm == "correlation":

        phaseID.voltage_correlation_transfo_ref()
        phaseID.plot_voltages(length=length)
        phaseID.plot_load_profiles(length=length)
        print("Accuracy using voltage correlation: ", phaseID.accuracy())
        print("wrong device ID's: ", phaseID.find_wrong_IDs())
