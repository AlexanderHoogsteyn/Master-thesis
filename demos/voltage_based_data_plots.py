from voltageBasedPhaseIdentification import *
import seaborn as sns

feeder = PhaseIdentification(measurement_error=0, feederID="86315_785383", include_three_phase=False, length=24*7)

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