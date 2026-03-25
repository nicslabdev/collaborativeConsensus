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

data_TH3_1 = data.loc[data["evModel"] == "MPNTAnomaly"]
data_TH3_2 = data.loc[data["evModel"] == "MPPTAnomaly"]
data2 = data.loc[data["evModel"] == "VolkswagenID4"]

y_TH3_1 = data_TH3_1.loc[:, ["duration"]].values / 3600
y_TH3_2 = data_TH3_2.loc[:, ["duration"]].values / 3600
y = data2.loc[:, ["duration"]].values / 3600

x_TH3_1 = data_TH3_1.loc[:, ["meanPower"]].values
x_TH3_2 = data_TH3_2.loc[:, ["meanPower"]].values
x = data2.loc[:, ["meanPower"]].values

# plt.xticks([0, 20, 40, 60, 80, 100])

plt.scatter(y_TH3_1, x_TH3_1, label="TH3 (under)", color="orange")
plt.scatter(y_TH3_2, x_TH3_2, label="TH3 (above)", color="red")
plt.scatter(y, x, label="Normal", color="blue")

plt.ylabel("Mean power (kW)")
plt.xlabel("Duration (hours)")
plt.title("TH3 (Mean Power-Duration)")
plt.legend()

#plt.show()
plt.savefig("th3.png")
