from __future__ import division
from __future__ import print_function

from ast import literal_eval

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyod.utils.data import check_consistent_shape
from pyod.utils.data import get_outliers_inliers
import scikitplot as skplt
from sklearn.metrics import average_precision_score, plot_precision_recall_curve
from matplotlib import rc
rc('mathtext', default='regular')

name = "chargingStationTransactions_ccs_100000_v3"
data = pd.read_csv("../datasets/chargingStationTransactions/chargingStationTransactions_ccs_100000_v3.csv")
data = data[:2000]

data_DNT = data.loc[data["evModel"] == "DNTAnomaly"]
data_DPT = data.loc[data["evModel"] == "DPTAnomaly"]
data2 = data.loc[data["evModel"] == "VolkswagenID4"]

y_DNT = data_DNT.loc[:, ["meanPower"]].values
y_DPT = data_DPT.loc[:, ["meanPower"]].values
y = data2.loc[:, ["meanPower"]].values

x_DNT = data_DNT.loc[:, ["duration"]].values / 3600
x_DPT = data_DPT.loc[:, ["duration"]].values / 3600
x = data2.loc[:, ["duration"]].values / 3600

# create figure and axis objects with subplots()
fig, ax = plt.subplots()
# make a plot
ax.scatter(x_DNT, y_DNT, color="red")
ax.scatter(x_DPT, y_DPT, label="TH1", color="red")
ax.scatter(x, y, color="blue")
ax.set_ylabel("Duration (hours)")

# TH2
data_TH2_1 = data.loc[data["evModel"] == "ECPTAnomaly"]
data_TH2_2 = data.loc[data["evModel"] == "ECNTAnomaly"]
data2 = data.loc[data["evModel"] == "VolkswagenID4"]

y_TH2_1 = data_TH2_1.loc[:, ["energyConsumption"]].values
y_TH2_2 = data_TH2_2.loc[:, ["energyConsumption"]].values
y = data2.loc[:, ["energyConsumption"]].values

x_TH2_1 = data_TH2_1.loc[:, ["duration"]].values / 3600
x_TH2_2 = data_TH2_2.loc[:, ["duration"]].values / 3600
x = data2.loc[:, ["duration"]].values / 3600

ax2 = ax.twinx()
ax2.scatter(x_TH2_1, y_TH2_1, color="orange")
ax2.scatter(x_TH2_2, y_TH2_2, label="TH2", color="orange")
ax2.scatter(x, y, label="energy no anomaly", color="black")
ax2.set_ylabel("Energy (kwh)")

# TH3

data_TH3_1 = data.loc[data["evModel"] == "MPNTAnomaly"]
data_TH3_2 = data.loc[data["evModel"] == "MPPTAnomaly"]
data2 = data.loc[data["evModel"] == "VolkswagenID4"]

y_TH3_1 = data_TH3_1.loc[:, ["meanPower"]].values
y_TH3_2 = data_TH3_2.loc[:, ["meanPower"]].values
y = data2.loc[:, ["meanPower"]].values

x_TH3_1 = data_TH3_1.loc[:, ["duration"]].values / 3600
x_TH3_2 = data_TH3_2.loc[:, ["duration"]].values / 3600
x = data2.loc[:, ["duration"]].values / 3600

# plt.xticks([0, 20, 40, 60, 80, 100])

ax.scatter(x_TH3_1, y_TH3_1, color="purple")
ax.scatter(x_TH3_2, y_TH3_2, label="TH3", color="purple")
ax.scatter(x, y, label="meanpower no anomaly", color="blue")

# added these three lines
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc=0)

# twin object for two different y-axis on the sample plot

plt.xlabel("Mean power (kW)")
plt.ylabel("Duration (hours)")
plt.title("TH1 in VolkswagenID4 (Duration-MeanPower)")

plt.show()
