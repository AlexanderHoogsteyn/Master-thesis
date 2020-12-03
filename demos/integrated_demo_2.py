from integratedPhaseIdentification import *
import seaborn as sns

"""
##################################################
DEMO 2
Influence of missing data on accuracy of voltage assisted load based method

I can improve this by making shure an additional 10 of missing is added in stead of all new devices
##################################################
"""
include_A = True
include_B = True
include_C = True
load_noise = 0.00   #pu
include_three_phase = False
length = 24*15
volt_assist = 0

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
        feeder = IntegratedMissingPhaseIdentification(measurement_error=load_noise, feederID=feeder_id,
                                                      include_three_phase=include_three_phase, length=15 * 24,
                                                      missing_ratio=0)
        for i, value in enumerate(missing_range):
            col = []
            for j, days in enumerate(length_range):
                feeder.reset_partial_phase_identification()
                feeder.reset_load_features_transfo()
                feeder.add_missing(value)
                feeder.voltage_assisted_load_correlation(sal_treshold_load=0.4, sal_treshold_volt=0.0, corr_treshold=0.1, volt_assist=volt_assist,length=24*days)
                col += [feeder.accuracy()]
            scores.append(col)
        tot_scores += np.array(scores)
        print(round(rep/reps*100), "% complete")
    tot_scores = tot_scores/reps
    print("TOTAL SCORE ", np.mean(tot_scores))

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
    plt.savefig("accuracy_low_sm_pen_integrated_"+feeder_id)
    plt.close()