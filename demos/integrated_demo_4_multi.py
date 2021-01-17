# import PhaseIdentification as pi
from src.PhaseIdentification.integratedPhaseIdentification import *
# from powerBasedPhaseIdentification import *
import pickle
import numpy as np
from multiprocessing import Pool
from itertools import product

"""
##################################################
DEMO 4
Influence of voltage assist ratio on accuracy of load based methods
but results get stored in pickle file

I can improve this by making shure an additional 10 of missing is added in stead of all new devices
##################################################
"""

load_noise = 0.01  # pu
include_three_phase = False
length = 24 * 15
cores = 4


def multiprocess(ratio, feeder_i):
    included_feeders = ["86315_785383", "65028_84566", "1830188_2181475"]

    length_range = np.arange(1, 15)
    missing_range = np.arange(0, 1.00, 0.10)
    ratio_range = np.arange(0, 1.1, 0.10)
    tot_scores = np.zeros([len(missing_range), len(length_range)])
    reps = 1

    for rep in range(0, reps):
        scores = []
        feeder = IntegratedMissingPhaseIdentification(measurement_error=load_noise, feederID=included_feeders[feeder_i],
                                                      include_three_phase=include_three_phase, length=15 * 24,
                                                      missing_ratio=0)
        for i, value in enumerate(missing_range):
            col = []
            for j, days in enumerate(length_range):
                feeder.reset_partial_phase_identification()
                feeder.reset_load_features_transfo()
                feeder.add_missing(value)
                feeder.voltage_assisted_load_correlation(sal_treshold_load=0.4, sal_treshold_volt=0.0,
                                                         corr_treshold=0.1, volt_assist=ratio_range[ratio],
                                                         length=24 * days)
                col += [feeder.accuracy()]
            scores.append(col)
        tot_scores += np.array(scores)
        # print(round(rep / reps * 100), "% complete")
    tot_scores = tot_scores / reps
    # print("TOTAL SCORE ", np.mean(tot_scores))

    return tot_scores


if __name__ == '__main__':
    included_feeders = ["86315_785383", "65028_84566", "1830188_2181475"]
    ratio_range = np.arange(0, 1.1, 0.1)
    data = {}
    configs = product(range(len(ratio_range)), range(len(included_feeders)))
    reps = 100
    length_range = np.arange(1, 15)
    missing_range = np.arange(0, 1.00, 0.10)

    # for g,feeder_id in enumerate(included_feeders):
    #    for h, ratio in enumerate(ratio_range):

    with Pool(processes=cores) as pool:
        scores = pool.starmap(multiprocess, configs)


    results = {"ratio_range": ratio_range, "length_range": length_range, "missing_range": missing_range,
               "reps": reps, "included_feeders": ["Case A","Case B", "Case C"], "data": dict(zip(product(range(len(ratio_range)), range(len(included_feeders))), scores))}

    with open('results_' + str(reps) + 'reps.pickle', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
