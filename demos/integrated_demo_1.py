import sys
from os.path import dirname
sys.path.append(dirname("../src/"))

from src.PhaseIdentification.voltageBasedPhaseIdentification import *
from src.PhaseIdentification.common import *
from src.PhaseIdentification.integratedPhaseIdentification import *

"""
##################################################
DEMO 1
Influence of missing data on accuracy of load based methods

I can improve this by making sure an additional 10 of missing is added in stead of all new devices
##################################################
"""

include_three_phase = True
length = 24*20
volt_assist = 0
reps = 1
sal_treshold = 0.1
corr_treshold = 0.6
salient_components = 4

#"1132967_1400879" only has 3 customers
# 'Case D'
included_feeders = ["86315_785383", "65028_84566", "1076069_1274129", "1132967_1400879", "65025_80035",
                    "1076069_1274125"]
results = []
for accuracy in [0.1, 0.2, 0.5, 1.0]:
    class_results = []
    for feeder_id in included_feeders:
        acc = 0
        for rep in np.arange(reps):
            feeder = Feeder(feederID=feeder_id, include_three_phase=include_three_phase)
            phase_identification = PartialPhaseIdentification(feeder, ErrorClass(accuracy, s=False))
            phase_identification.load_correlation(salient_treshold=sal_treshold, corr_treshold=corr_treshold, salient_components=salient_components, length=24*20)
            acc = acc + phase_identification.accuracy()
        class_results.append(acc*100/reps)
        print("Feeder: ", feeder_id)
    results.append(class_results)

for accuracy_s in [0.2, 0.5]:
    class_results = []
    for feeder_id in included_feeders:
        acc = 0
        for rep in np.arange(reps):
            feeder = Feeder(feederID=feeder_id, include_three_phase=include_three_phase)
            phase_identification = PartialPhaseIdentification(feeder, ErrorClass(accuracy_s, s=True))
            phase_identification.load_correlation(salient_treshold=sal_treshold, corr_treshold=corr_treshold, salient_components=salient_components, length=24*20)
            acc = acc + phase_identification.accuracy()
        class_results.append(acc*100/reps)
        print("Feeder: ", feeder_id)
    results.append(class_results)


labels = ['Case A', 'Case B', 'Case C','Case D', 'Case E', 'Case F']

x = np.arange(len(labels))  # the label locations
width = 0.15  # the width of the bars

fig, ax = plt.subplots(figsize=(9,3))
rects1 = ax.bar(x - width*5/2, results[0], width, label='Class 0.1')
rects2 = ax.bar(x - width*3/2, results[1], width, label='Class 0.2')
rects3 = ax.bar(x - width/2, results[2], width, label='Class 0.5')
rects4 = ax.bar(x + width/2, results[3], width, label='Class 1.0')
rects5 = ax.bar(x + width*3/2, results[4], width, label='Class 0.2s')
rects6 = ax.bar(x + width*5/2, results[5], width, label='Class 0.5s')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Accuracy (%)')
#ax.set_title('Accuracy by case and accuracy class using voltage Pearson correlation')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(bbox_to_anchor=(1,0.0),loc='lower right',ncol=2)


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 4, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


#autolabel(rects1)
#autolabel(rects2)
#autolabel(rects3)
#autolabel(rects4)

fig.tight_layout()

plt.show()