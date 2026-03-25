
from __future__ import division
from __future__ import print_function

from ast import literal_eval

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv("../datasets/chargingStationTransactions/chargingStationTransactions_ccs_1000_meterValues.csv")
features = ['meterValues']

data_PNSA = data.loc[data["evModel"] == "PNSAnomaly"]
data_PPSA = data.loc[data["evModel"] == "PPSAnomaly"]
data_VWID4 = data.loc[data["evModel"] == "VolkswagenID4"]
y_PNSA = data_PNSA.loc[:, features].values
y_VWID4 = data_VWID4.loc[:, features].values
y_PPSA = data_PPSA.loc[:, features].values

y_PNSA = y_PNSA[2][0]
y_PNSA = literal_eval(y_PNSA)
y_PPSA = y_PPSA[5][0]
y_PPSA = literal_eval(y_PPSA)
y_VWID4 = y_VWID4[0][0]
y_VWID4 = literal_eval(y_VWID4)

x_VWID4 = np.arange(1, 100, 100/len(y_VWID4))
plt.plot(x_VWID4, y_VWID4, label="Normal" )

x_PNSA = np.arange(1, 100, 100/len(y_PNSA))
plt.plot(x_PNSA, y_PNSA, label="TH5 (under)", color="orange")

x_PPSA = np.arange(1, 100, 100/len(y_PPSA))
plt.plot(x_PPSA, y_PPSA, label="TH5 (above)", color="red")

plt.xticks([0, 20, 40, 60, 80, 100])

plt.xlabel("Battery (SoC)")
plt.ylabel("Charge Power (kW)")
plt.title("TH5 (Volkswagen ID.4 1st: 126 kW)")
plt.legend()

#plt.show()
plt.savefig("th5.png")