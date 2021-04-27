import sys
from os.path import dirname
sys.path.append(dirname("../src/"))

from PhaseIdentification.powerBasedPhaseIdentification import *

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
include_three_phase = True
length = 24*20
accuracy_class = 0.1
reps = 100


included_feeders = ["1351982_1596442"]

cases = ["Case D"]


for i, feeder_id in enumerate(included_feeders):
    acc = 0
    for rep in range(reps):
        feeder = Feeder(feederID=feeder_id, include_three_phase=include_three_phase)
        phase_identification = PartialPhaseIdentification(feeder, ErrorClass(accuracy_class))
        #phase_identification.load_correlation_xu_fixed(nb_salient_components=440, length=length, salient_components=1)
        phase_identification.load_correlation_xu_fixed(nb_salient_components=400,salient_components=1)
        acc = acc + phase_identification.accuracy()
    print(cases[i], " acc uracy = ", 100*acc/reps)