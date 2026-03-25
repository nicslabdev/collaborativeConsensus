
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

name = "chargingStationTransactions_ccs_1000_meterValues"
data = pd.read_csv("../datasets/chargingStationTransactions/chargingStationTransactions_ccs_1000_meterValues.csv")
features = ['meterValues']

data_DNT = data.loc[data["evModel"] == "DNTAnomaly"]
data_DPT = data.loc[data["evModel"] == "DPTAnomaly"]
data2 = data.loc[data["evModel"] == "VolkswagenID4"]

y_DNT = data_DNT.loc[:, ["duration"]].values / 3600
y_DPT = data_DPT.loc[:, ["duration"]].values / 3600
y = data2.loc[:, ["duration"]].values / 3600

x_DNT = data_DNT.loc[:, ["meanPower"]].values
x_DPT = data_DPT.loc[:, ["meanPower"]].values
x = data2.loc[:, ["meanPower"]].values

#plt.xticks([0, 20, 40, 60, 80, 100])

plt.scatter(x_DNT, y_DNT, label="TH1 (under)", color = "orange")
plt.scatter(x_DPT, y_DPT, label="TH1 (above)", color = "red")
plt.scatter(x, y, label="Normal", color = "blue")

plt.xlabel("Mean power (kW)")
plt.ylabel("Duration (hours)")
plt.title("TH1 (Duration-MeanPower)")
plt.legend()

#plt.show()
plt.savefig("th1.png")