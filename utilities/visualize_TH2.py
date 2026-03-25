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

name = "chargingStationTransactions_ccs_100000_v3"
data = pd.read_csv("../datasets/chargingStationTransactions/chargingStationTransactions_ccs_100000_v3.csv")
data = data[:2000]

data_TH2_1 = data.loc[data["evModel"] == "ECPTAnomaly"]
data_TH2_2 = data.loc[data["evModel"] == "ECNTAnomaly"]
data2 = data.loc[data["evModel"] == "VolkswagenID4"]

y_TH2_1 = data_TH2_1.loc[:, ["energyConsumption"]].values
y_TH2_2 = data_TH2_2.loc[:, ["energyConsumption"]].values
y = data2.loc[:, ["energyConsumption"]].values

x_TH2_1 = data_TH2_1.loc[:, ["duration"]].values / 3600
x_TH2_2 = data_TH2_2.loc[:, ["duration"]].values / 3600
x = data2.loc[:, ["duration"]].values / 3600

plt.scatter(x_TH2_2, y_TH2_2, label="TH2 (under)", color="orange")
plt.scatter(x_TH2_1, y_TH2_1, label="TH2 (above)", color="red")
plt.scatter(x, y, label="Normal", color="blue")

plt.xlabel("Duration (hours)")
plt.ylabel("Energy (kwh)")
plt.title("TH2 (Energy-MeanPower)")
plt.legend()

#plt.show()
plt.savefig("th2.png")
