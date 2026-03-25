
from __future__ import division
from __future__ import print_function

from ast import literal_eval

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv("../datasets/chargingStationTransactions/chargingStationTransactions_full_meterValues.csv")
features = ['meterValues']

data_normal = data.loc[data["anomaly"] == 0]

data_schuko = data_normal.loc[data_normal["typeSocket"] == "F"]
data_mennekes = data_normal.loc[data_normal["typeSocket"] == "2"]
data_ccs = data_normal.loc[data_normal["typeSocket"] == "CCS"]

data_mennekes_wallplug = data_mennekes.loc[(2.1 <= data_mennekes["meanPower"]) & (data_mennekes["meanPower"] <= 2.5)]
data_mennekes_onephase16 = data_mennekes.loc[(3.5 <= data_mennekes["meanPower"]) & (data_mennekes["meanPower"] <= 3.9)]
data_mennekes_onephase32 = data_mennekes.loc[(7.2 <= data_mennekes["meanPower"]) & (data_mennekes["meanPower"] <= 7.6)]
data_mennekes_threephase16 = data_mennekes.loc[(10.8 <= data_mennekes["meanPower"]) & (data_mennekes["meanPower"] <= 11.2)]
data_mennekes_threephase32 = data_mennekes.loc[(21.8 <= data_mennekes["meanPower"]) & (data_mennekes["meanPower"] <= 22.2)]

data_ccs_hondae = data_ccs.loc[data_ccs["evModel"] == "HondaE"]
data_ccs_citroenec4 = data_ccs.loc[data_ccs["evModel"] == "CitroenEC4"]
data_ccs_vwid4 = data_ccs.loc[data_ccs["evModel"] == "VolkswagenID4"]
data_ccs_teslamodel3 = data_ccs.loc[data_ccs["evModel"] == "TeslaModel3LongRange"]
data_ccs_porsche = data_ccs.loc[data_ccs["evModel"] == "PorscheTaycanTurboS"]
data_ccs_audi = data_ccs.loc[data_ccs["evModel"] == "AudiETronGTQuattro"]

y_SCHUKO = data_schuko.loc[:, features].values
y_MENNEKES_WallPlug = data_mennekes_wallplug.loc[:, features].values
y_MENNEKES_ONEPHASE16 = data_mennekes_onephase16.loc[:, features].values
y_MENNEKES_ONEPHASE32 = data_mennekes_onephase32.loc[:, features].values
y_MENNEKES_THREEPHASE16 = data_mennekes_threephase16.loc[:, features].values
y_MENNEKES_THREEPHASE32 = data_mennekes_threephase32.loc[:, features].values
y_CCS_HONDAE = data_ccs_hondae.loc[:, features].values
y_CCS_CITROENEC4 = data_ccs_citroenec4.loc[:, features].values
y_CCS_VWID4 = data_ccs_vwid4.loc[:, features].values
y_CCS_TESLA = data_ccs_teslamodel3.loc[:, features].values
y_CCS_PORSCHE = data_ccs_porsche.loc[:, features].values
y_CCS_AUDI = data_ccs_audi.loc[:, features].values

y_SCHUKO = literal_eval(y_SCHUKO[1][0])
y_MENNEKES_WallPlug = literal_eval(y_MENNEKES_WallPlug[0][0])
y_MENNEKES_ONEPHASE16 = literal_eval(y_MENNEKES_ONEPHASE16[0][0])
y_MENNEKES_ONEPHASE32 = literal_eval(y_MENNEKES_ONEPHASE32[0][0])
y_MENNEKES_THREEPHASE16 = literal_eval(y_MENNEKES_THREEPHASE16[0][0])
y_MENNEKES_THREEPHASE32 = literal_eval(y_MENNEKES_THREEPHASE32[0][0])
y_CCS_HONDAE = literal_eval(y_CCS_HONDAE[0][0])
y_CCS_CITROENEC4 = literal_eval(y_CCS_CITROENEC4[0][0])
y_CCS_VWID4 = literal_eval(y_CCS_VWID4[0][0])
y_CCS_TESLA = literal_eval(y_CCS_TESLA[0][0])
y_CCS_PORSCHE = literal_eval(y_CCS_PORSCHE[0][0])
y_CCS_AUDI = literal_eval(y_CCS_AUDI[0][0])

x_SCHUKO = np.arange(0, len(y_SCHUKO))/len(y_SCHUKO)*100
# plt.plot(x_SCHUKO, y_SCHUKO, label="Schuko", color="blue")
x_MENNEKES_WallPlug = np.arange(0, len(y_MENNEKES_WallPlug))/len(y_MENNEKES_WallPlug)*100
x_MENNEKES_ONEPHASE16 = np.arange(0, len(y_MENNEKES_ONEPHASE16))/len(y_MENNEKES_ONEPHASE16)*100

plt.fill(np.append(x_MENNEKES_WallPlug, x_MENNEKES_ONEPHASE16[::-1]),
         np.append(y_MENNEKES_WallPlug, y_MENNEKES_ONEPHASE16[::-1]),
         alpha=0.5, label="Schuko (2.3 kW - 3.7 kW)")

plt.plot(x_MENNEKES_WallPlug, y_MENNEKES_WallPlug, label="Mennekes (Wall Plug: 2.3 kW)", linewidth=2)

plt.plot(x_MENNEKES_ONEPHASE16, y_MENNEKES_ONEPHASE16, label="Mennekes (1-Phase 16A: 3.7 kW)", linewidth=2)

x_MENNEKES_ONEPHASE32 = np.arange(0, len(y_MENNEKES_ONEPHASE32))/len(y_MENNEKES_ONEPHASE32)*100
plt.plot(x_MENNEKES_ONEPHASE32, y_MENNEKES_ONEPHASE32, label="Mennekes (1-Phase 32A: 7.4 kW)")

x_MENNEKES_THREEPHASE16 = np.arange(0, len(y_MENNEKES_THREEPHASE16))/len(y_MENNEKES_THREEPHASE16)*100
plt.plot(x_MENNEKES_THREEPHASE16, y_MENNEKES_THREEPHASE16, label="Mennekes (3-Phase 16A: 11 kW)")

x_MENNEKES_THREEPHASE32 = np.arange(0, len(y_MENNEKES_THREEPHASE32))/len(y_MENNEKES_THREEPHASE32)*100
plt.plot(x_MENNEKES_THREEPHASE32, y_MENNEKES_THREEPHASE32, label="Mennekes (3-Phase 32A: 22 kW)")

# x_CCS_HONDAE = np.arange(0, len(y_CCS_HONDAE))/len(y_CCS_HONDAE)*100
# plt.plot(x_CCS_HONDAE, y_CCS_HONDAE, label="Honda E (peak: 56 kW)")
#
# x_CCS_CITROENEC4 = np.arange(0, len(y_CCS_CITROENEC4))/len(y_CCS_CITROENEC4)*100
# plt.plot(x_CCS_CITROENEC4, y_CCS_CITROENEC4, label="Citröen E-C4 (peak: 100 kW)")
#
# x_CCS_VWID4 = np.arange(0, len(y_CCS_VWID4))/len(y_CCS_VWID4)*100
# plt.plot(x_CCS_VWID4, y_CCS_VWID4, label="Volkswagen ID.4 1st (peak: 126 kW)")
#
# x_CCS_TESLA = np.arange(0, len(y_CCS_TESLA))/len(y_CCS_TESLA)*100
# plt.plot(x_CCS_TESLA, y_CCS_TESLA, label="Tesla Model 3 Long Range (peak: 148 kW)")
#
# x_CCS_PORSCHE = np.arange(0, len(y_CCS_PORSCHE))/len(y_CCS_PORSCHE)*100
# plt.plot(x_CCS_PORSCHE, y_CCS_PORSCHE, label="Porsche Taycan Turbo S (peak: 166 kW)")
#
# x_CCS_AUDI = np.arange(0, len(y_CCS_AUDI))/len(y_CCS_AUDI)*100
# plt.plot(x_CCS_AUDI, y_CCS_AUDI, label="Audi GT Quattro (peak: 175 kW)")

plt.xticks([0, 20, 40, 60, 80, 100])

plt.xlabel("Battery (%) (SoC)")
plt.ylabel("Charging Power (kW)")
plt.title("Slow Charging Curves (Schuko & Mennekes)", fontweight='bold')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.14),
          fancybox=True, shadow=False, ncol=2)
plt.tight_layout()
#plt.show()
plt.savefig("Slow_Charging_Curves_Figure.png")