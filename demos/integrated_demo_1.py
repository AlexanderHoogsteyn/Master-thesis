from integratedPhaseIdentification import *
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
load_noise = 0.0   #pu
include_three_phase = False
length = 24*20
volt_assist = 0.0

included_feeders = []
if include_A:
    included_feeders.append("86315_785383")
if include_B:
    included_feeders.append("65028_84566")
if include_C:
    included_feeders.append("1830188_2181475")

for feeder_id in included_feeders:
    load_feeder = IntegratedPhaseIdentification(measurement_error=load_noise, feederID=feeder_id,
                    include_three_phase=include_three_phase, length=length)
    load_feeder.voltage_assisted_load_correlation(sal_treshold_load=0.4, sal_treshold_volt=0.0, corr_treshold=0, volt_assist=volt_assist)
    acc = load_feeder.accuracy()
    print("Accuracy", acc*100, "%")
