import sys
from os.path import dirname
sys.path.append(dirname("../src/"))

from src.PhaseIdentification.voltageBasedPhaseIdentification import *

include_three_phase = True
length = 24*20*1
accuracy_class = 0.5
reps = 1
nb_salient_components = int(length)

included_feeders = ["86315_785383", "1076069_1274129", "1351982_1596442", "65025_80035"]
cases = ["Case A","Case C","Case D","Case E"]
for i, feeder_id in enumerate(included_feeders):
    acc = 0
    for rep in range(reps):
        feeder = Feeder(feederID=feeder_id, include_three_phase=include_three_phase)
        phase_identification = PhaseIdentification(feeder, ErrorClass(accuracy_class,s=True))
        #phase_identification.load_correlation_xu_fixed(nb_salient_components=440, length=length, salient_components=1)
        phase_identification.voltage_correlation_transfo_ref(length=length)
        acc = acc + phase_identification.accuracy()
    print(cases[i], " accuracy = ", 100*acc/reps)