
from __future__ import division
from __future__ import print_function

from ast import literal_eval

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

data = pd.read_csv("../datasets/chargingStationStates/chargingStationStates_ccs_100000.csv")
#data['status'] = data['status'].apply(lambda x: 1 if x == 'active' else 0).astype(int)

data_TH = data.loc[data["evModel"] == "ConsumptionInactiveStateAnomaly"]
data_TH = data_TH[500:600]
data2 = data.loc[data["anomaly"] == 0]
data2 = data2[5000:5100]

y_TH = data_TH.loc[:, ["status"]].values
y = data2.loc[:, ["status"]].values

x_TH = data_TH.loc[:, ["power"]].values
x = data2.loc[:, ["power"]].values

# plt.xticks([0, 20, 40, 60, 80, 100])

#plt.scatter(x_TH, y_TH, label="TH6", color="red")
#plt.scatter(x, y, label="No anomaly", color="blue")

data3 = pd.concat([data2, data_TH])
ax = sns.stripplot(x="power", y="status", data=data2, color="blue", label="no anomaly")
sns.stripplot(x="power", y="status", data=data_TH, color="red", ax=ax, label="TH6")

plt.xlabel("Power (kW)")
plt.ylabel("Status")
plt.title("TH6 (Status-Power)")
plt.legend()

plt.show()