import sys
from os.path import dirname
sys.path.append(dirname("../src/"))

from PhaseIdentification.integratedPhaseIdentification import IntegratedPhaseIdentification

"""
##################################################
DEMO 1
Influence of missing data on accuracy of load based methods

I can improve this by making shure an additional 10 of missing is added in stead of all new devices
##################################################
"""

load_noise = 0.0#pu
include_three_phase = False
length = 24*20
volt_assist = 0.9

included_feeders = ["86315_785383", "65028_84566", "1076069_1274129", "1132967_1400879", "65025_80035", "1076069_1274125"]

for feeder_id in included_feeders:
    feeder = IntegratedPhaseIdentification(measurement_error=load_noise, feederID=feeder_id,
                    include_three_phase=include_three_phase, length=length)
    feeder.voltage_assisted_load_correlation(sal_treshold_load=1, sal_treshold_volt=0.0, corr_treshold=2, volt_assist= volt_assist, length=length,salient_components=4,printout_level=0)
    #feeder.plot_voltages(length=length)
    #feeder.plot_load_profiles(length=length)

    #C = CorrelationCoeficients(feeder)
    #C.visualize_correlation_all()