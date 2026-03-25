
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

data_MPPT = data.loc[data["evModel"] == "MPPTAnomaly"]
data_MPNT = data.loc[data["evModel"] == "MPNTAnomaly"]
data_CEC4 = data.loc[data["evModel"] == "CitroenEC4"]

y_MPPT = data_MPPT.loc[:, ["meanPower"]].values
y_MPNT = data_MPNT.loc[:, ["meanPower"]].values
y = data_CEC4.loc[:, features].values
y_CEC4 = data_CEC4.loc[:, ["meanPower"]].values

y = y[0][0]
y = literal_eval(y)
y_MPPT = [y_MPPT[2]] * len(y)
y_MPNT = [y_MPNT[1]] * len(y)
y_CEC4 = [y_CEC4[0]] * len(y)

x_CEC4 = np.arange(1, 100, 100/len(y_CEC4))
plt.plot(x_CEC4, y)

x_CEC4 = np.arange(1, 100, 100/len(y_CEC4))
plt.plot(x_CEC4, y_CEC4, label="Normal (Mean: "+str(round(y_CEC4[0][0], 2))+" kW)")

x_MPNT = np.arange(1, 100, 100/len(y_MPNT))
plt.plot(x_MPNT, y_MPNT, label="MPNTAnomaly (Mean: "+str(round(y_MPNT[0][0], 2))+" kW)")

x_MPPT = np.arange(1, 100, 100/len(y_MPPT))
plt.plot(x_MPPT, y_MPPT, label="MPPTAnomaly (Mean: "+str(round(y_MPPT[0][0], 2))+" kW)")

plt.xticks([0, 20, 40, 60, 80, 100])

plt.xlabel("Battery (SoC)")
plt.ylabel("Charge Power (kW)")
plt.title("Citröen EC-4 (100 kW)")
plt.legend()

plt.show()