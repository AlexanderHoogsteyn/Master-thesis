from src.PhaseIdentification.powerBasedPhaseIdentification import *
import seaborn as sns

"""
##################################################
DEMO 3
Influence of missing data on accuracy of load based methods

I can improve this by making shure an additional 10 of missing is added in stead of all new devices
##################################################
"""
include_A = True
include_B = True
include_C = True
load_noise = 0.01   #pu
include_three_phase = False
length = 24*7


included_feeders = []
if include_A:
    included_feeders.append("86315_785383")
if include_B:
    included_feeders.append("65028_84566")
if include_C:
    included_feeders.append("1830188_2181475")

for feeder_id in included_feeders:
    length_range = np.arange(1, 15)
    missing_range = np.arange(0, 1.00, 0.10)
    tot_scores = np.zeros([len(missing_range), len(length_range)])
    reps = 1
    for rep in range(0,reps):
        scores = []
        for i, value in enumerate(missing_range):
            col = []
            for j, length in enumerate(length_range):
                load_feeder = PartialMissingPhaseIdentification(measurement_error=load_noise, feederID=feeder_id,
                                                                include_three_phase=include_three_phase, length=length*24,
                                                                missing_ratio=value)
                load_feeder.load_correlation(sal_treshold=0.4, corr_treshold=-np.inf)
                col += [load_feeder.accuracy()]
            scores.append(col)
        tot_scores += np.array(scores)
        print(round(rep/reps*100), "% complete")
    tot_scores = tot_scores/reps
    # Plot
    plt.figure(figsize=(12, 10), dpi=80)
    y = [str(i) + "%" for i in list(np.arange(100, 0, -10))]
    x = length_range
    sns.heatmap(tot_scores, xticklabels=x, yticklabels=y, cmap='RdYlGn', center=0.7,
                annot=True)

    # Decorations
    plt.title('Percentage of customers that are allocated correctly for ' + str(feeder_id), fontsize=16)
    plt.xticks(fontsize=12)
    plt.xlabel("Duration (days) that hourly data was collected")
    plt.ylabel("Percentage of customers with smart meter")
    plt.yticks(fontsize=12)
    plt.show()