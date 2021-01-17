import sys
from os.path import dirname
sys.path.append(dirname("../src/"))

from PhaseIdentification.integratedPhaseIdentification import *
import pickle
import numpy as np

"""
##################################################
DEMO 4
Influence of voltage assist ratio on accuracy of load based methods
but results get stored in pickle file

I can improve this by making shure an additional 10 of missing is added in stead of all new devices
##################################################
"""
load_noise = 0.01   #pu
include_three_phase = False
length = 24*15

included_feeders = ["86315_785383", "65028_84566", "1076069_1274129", "1132967_1400879", "65025_80035", "1076069_1274125"]

ratio_range = np.arange(0,1.2,0.20)
length_range = np.arange(1, 15)
missing_range = np.arange(0, 1.00, 0.10)
reps = 1

data = {}

for g,feeder_id in enumerate(included_feeders):
    for h, ratio in enumerate(ratio_range):
        tot_scores = np.zeros([len(missing_range), len(length_range)])
        for rep in range(0, reps):
            scores = []
            feeder = IntegratedMissingPhaseIdentification(measurement_error=load_noise, feederID=feeder_id,
                                                          include_three_phase=include_three_phase, length=15 * 24,
                                                          missing_ratio=0)
            for i, value in enumerate(missing_range):
                col = []
                for j, days in enumerate(length_range):
                    feeder.reset_partial_phase_identification()
                    feeder.reset_load_features_transfo()
                    feeder.add_missing(value)
                    feeder.voltage_assisted_load_correlation(sal_treshold_load=0.4, sal_treshold_volt=0.0,
                                                             corr_treshold=0.1, volt_assist=ratio,
                                                             length=24 * days, printout_level=1)
                    col += [feeder.accuracy()]
                scores.append(col)
            tot_scores += np.array(scores)
            #print(round(rep / reps * 100), "% complete")
        tot_scores = tot_scores / reps
        #print("TOTAL SCORE ", np.mean(tot_scores))

        data[(h,g)] = tot_scores

results = {"ratio_range":ratio_range,"length_range":length_range,"missing_range":missing_range,
           "reps":reps, "included_feeders":["Case A"],"data":data}

with open('results_'+str(reps)+'reps_local.pickle', 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)