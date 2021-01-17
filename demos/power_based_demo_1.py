from src.PhaseIdentification.powerBasedPhaseIdentification import *

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
Still some inaccuracies? Where does this come from -> Empty load profiles
"""
include_A = True
include_B = True
include_C = True
load_noise = 0.0   #pu
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

    load_feeder = PartialPhaseIdentification(measurement_error=load_noise, feederID=feeder_id,
                                             include_three_phase=include_three_phase)
    print("Start load correlation algorithm for ", feeder_id)
    load_feeder.load_correlation(sal_treshold=0.1, corr_treshold=0.0)
