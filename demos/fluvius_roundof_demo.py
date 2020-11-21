from voltageBasedPhaseIdentification import *

"""
##################################################
Fluvinus experiment can be done using "truncated"
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
voltage_noise = 0.00
load_noise = 0.03   #pu
include_three_phase = False
length = 24*7
n_repeats = 1


feeder.truncate_voltages()

feeder.k_means_clustering(3, n_repeats=n_repeats)
print("Accuracy using ", representation, " data: ", feeder.accuracy())
feeder.plot_voltages(length=length)
feeder.plot_load_profiles(length=length)