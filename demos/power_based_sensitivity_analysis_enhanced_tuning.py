import sys
from os.path import dirname
sys.path.append(dirname("../src/"))

from PhaseIdentification.powerBasedPhaseIdentification import *
from PhaseIdentification.common import *
import seaborn as sns

"""
##################################################
DEMO 3
Influence of voltage assist ratio on accuracy of load based methods
For multiple feeders

I can improve this by making shure an additional 10 of missing is added in stead of all new devices
##################################################
"""

included_feeders = ["86315_785383", "65028_84566", "1076069_1274129","1132967_1400879", "65025_80035", "1076069_1274125"]
cases = ["Case A","Case B","Case C","Case D","Case E","Case F"]
include_three_phase = True
length = 24*14
salient_components = 4
reps = 20
accuracy = 0.1

for case, feeder_id in enumerate(included_feeders):
    nb_assigments_range = [1]#np.arange(1, 11, 5)
    salient_comp_range = [10]#np.arange(1, 11, 5)
    tot_scores = np.zeros([len(salient_comp_range), len(nb_assigments_range)])

    for rep in range(0,reps):
        scores = []
        for i, sal_comp in enumerate(salient_comp_range):
            col = []
            for j, nb_assignments in enumerate(nb_assigments_range):
                feeder = Feeder(feederID=feeder_id, include_three_phase=include_three_phase)
                phase_identification = PartialPhaseIdentification(feeder, ErrorClass(accuracy, s=False))
                phase_identification.load_correlation_enhanced_tuning(nb_assignments=nb_assignments,nb_salient_components=sal_comp, length=length)

                col += [phase_identification.accuracy()]
            scores.append(col)
        tot_scores += np.array(scores)
        print(round(rep/reps*100), "% complete")
    tot_scores = tot_scores/reps
    # Plot
    plt.figure(figsize=(12, 10), dpi=80)
    y = ["%.2f" % i for i in list(salient_comp_range)]
    x = ["%.2f" % i for i in list(salient_comp_range)]
    sns.heatmap(tot_scores, xticklabels=x, yticklabels=y, cmap='RdYlGn', center=0.7,
                annot=True,cbar=False)

    # Decorations
    plt.title(cases[case], fontsize=16)
    plt.xticks(fontsize=12)
    plt.xlabel("Assigments / iteration")
    plt.ylabel("Salient components")
    plt.yticks(fontsize=12)
    plt.show()



include_three_phase = False
length = 24*20
volt_assist = 0
reps = 1
sal_treshold = 0.001
corr_treshold = 0.3
