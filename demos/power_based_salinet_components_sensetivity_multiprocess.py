import sys
from os.path import dirname
sys.path.append(dirname("../src/"))

from PhaseIdentification.powerBasedPhaseIdentification import *
from PhaseIdentification.common import *
import matplotlib.pyplot as plt
import multiprocessing

"""
##################################################
DEMO 3
Influence of voltage assist ratio on accuracy of load based methods
For multiple feeders

I can improve this by making shure an additional 10 of missing is added in stead of all new devices
#################################################
"""

def worker(feeder_id):
    include_three_phase = True
    length = 48 * 360
    salient_components = 1
    accuracy = 0.1
    sal_treshold = 2
    acc_class_range = np.array([0.1,0.2,0.5,1.0,0.2,0.5])
    sal_treshold_range = list(range(4,48*360,150))
    #voltage_pen_range = np.array([0.1])
    s_range = [False,False,False,False,True,True]
    scores = []
    for i, accuracy in enumerate(acc_class_range):
        col = []
        for j, nb_salient_components in enumerate(sal_treshold_range):
            feeder = Feeder(feederID=feeder_id, include_three_phase=include_three_phase)
            phase_identification = PartialPhaseIdentification(feeder, ErrorClass(accuracy, s=s_range[i]))
            phase_identification.load_correlation_xu_fixed(nb_salient_components=nb_salient_components,
                                                  salient_components=salient_components, length=length)

            col += [phase_identification.accuracy()]
        scores.append(col)
    return np.array(scores)


if __name__ == '__main__':
    included_feeders = ["86315_785383", "1076069_1274129", "1351982_1596442", "65025_80035"]
    cases = ["Case A", "Case C", "Case D", "Case E", ]
    sal_treshold_range = list(range(4,48*360,150))
    reps = 2
    jobs = []
    for case, feeder_id in enumerate(included_feeders):
        data = [feeder_id]*reps
        p = multiprocessing.Pool(40)
        scores = p.map(worker,data)
        tot_scores = np.zeros([len(acc_class_range), len(sal_treshold_range)])
        for i in scores:
            tot_scores = tot_scores + i
        tot_scores = tot_scores/reps

        plt.figure(figsize=(8, 6), dpi=80)
        y = sal_treshold_range
        x = ["Class 0.1","Class 0.2","Class 0.5","Class 1.0","Class 0.2s","Class 0.5s"]
        #x = ["Class 0.1"]
        for i,c in enumerate(x):
            plt.plot(y, tot_scores[i]*100, label=c)

        # Decorations
        plt.rc('font', size=14)
        plt.title(cases[case], fontsize=20)
        plt.xticks(fontsize=12)
        plt.ylim([25,105])
        plt.xlabel("Number of salient components",fontsize=20)
        plt.ylabel("Accuracy (%)",fontsize=20)
        plt.yticks(fontsize=12)
        plt.legend()
        #plt.show()
        plt.savefig("salient_components_sensetivity_delta_" + cases[case])

