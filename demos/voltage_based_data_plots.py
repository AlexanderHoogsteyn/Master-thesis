from voltageBasedPhaseIdentification import *
from integratedPhaseIdentification import *
import seaborn as sns

feeder = PhaseIdentification(measurement_error=0, feederID="1830188_2181475", include_three_phase=False, length=24*7)

voltage_data = feeder.voltage_features
plt.figure(figsize=(6, 3))
markers = ["s", "o", "D", ">", "<", "v", "+"]
x = np.arange(0, 24*7)
color = plt.cm.viridis(float(1) / (float(feeder.nb_customers) - 1.0))
plt.plot(x, voltage_data[1], color=color, alpha=0.85)
plt.yticks(fontsize=10)
plt.ylabel("Voltage", fontsize=14)
plt.xlabel("Time", fontsize=14)
plt.show()

plt.figure(figsize=(6, 3))
markers = ["s", "o", "D", ">", "<", "v", "+"]
x = np.arange(0, 24*7)
color = plt.cm.viridis(float(8) / (float(feeder.nb_customers) - 1.0))
plt.plot(x, voltage_data[8], color=color, alpha=0.85)
plt.yticks(fontsize=10)
plt.ylabel("Voltage", fontsize=14)
plt.xlabel("Time", fontsize=14)
plt.show()

plt.figure(figsize=(6, 3))
markers = ["s", "o", "D", ">", "<", "v", "+"]
x = np.arange(0, 24*7)
color = plt.cm.viridis(float(13) / (float(feeder.nb_customers) - 1.0))
plt.plot(x, voltage_data[13], color=color, alpha=0.85)
plt.yticks(fontsize=10)
plt.ylabel("Voltage", fontsize=14)
plt.xlabel("Time", fontsize=14)
plt.show()

transfo_volt_data = feeder._voltage_features_transfo
plt.figure(figsize=(6, 3))
x = np.arange(0, 24*7)
color = plt.cm.viridis(float(13) / (float(feeder.nb_customers) - 1.0))
plt.plot(x, transfo_volt_data[0], color=color, alpha=0.85)
plt.plot(x, transfo_volt_data[1], color=color, alpha=0.85)
plt.plot(x, transfo_volt_data[2], color=color, alpha=0.85)
plt.yticks(fontsize=10)
plt.ylabel("Voltage", fontsize=14)
plt.xlabel("Time", fontsize=14)
plt.show()