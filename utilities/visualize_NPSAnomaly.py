
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
data = pd.read_csv("../datasets/chargingStationStates/chargingStationStates")
features = ['meterValues']

data_NPSA = data.loc[data["evModel"] == "NPSAnomaly"]
data_CEC4 = data.loc[data["evModel"] == "CitroenEC4"]

y_NPSA = data_NPSA.loc[:, features].values
y_CEC4 = data_CEC4.loc[:, features].values

y_NPSA = y_NPSA[9][0]
y_NPSA = literal_eval(y_NPSA)
y_CEC4 = y_CEC4[0][0]
y_CEC4 = literal_eval(y_CEC4)

x_CEC4 = np.arange(1, 100, 100/len(y_CEC4))
plt.plot(x_CEC4, y_CEC4, label="Normal" )

x_NPSA = np.arange(1, 100, 100/len(y_NPSA))
plt.plot(x_NPSA, y_NPSA, label="TH4")

plt.xticks([0, 20, 40, 60, 80, 100])

plt.xlabel("Battery (SoC)")
plt.ylabel("Charge Power (kW)")
plt.title("Citröen EC-4 (100 kW)")
plt.legend()

plt.show()