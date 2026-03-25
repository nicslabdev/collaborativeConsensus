from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from yellowbrick.features.pcoords import parallel_coordinates
from yellowbrick.model_selection import ValidationCurve
import pandas as pd
import numpy as np

from anomaliesDetection.utilities.functions import encode_variable

name = "chargingStationStates_full_200000_WithPCA"
data = pd.read_csv("../datasets/chargingStationStates/chargingStationStates_full_200000.csv")

features = ['power', 'status', 'idUser', 'typeSocket']
use_pca = True
percentage_test = 0.3
n_components_pca = 2
contamination = 0.01

# Encode strings to float
data['status'] = data['status'].apply(lambda x: 1 if x == 'active' else 0).astype(int)
data['idUser'] = data['idUser'].apply(lambda x: 1 if isinstance(x, str) else 0).astype(int)
encode_variable(data, data.columns.get_loc("typeSocket"))

# separate data: train and test
x = data.loc[:, features].values
y = data.loc[:, ['anomaly']].values.ravel()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=percentage_test)

# Standarization
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# Specify the features of interest and the classes of the target
classes = ["normal", "anomaly"]

# Instantiate the visualizer
#visualizer = parallel_coordinates(x, y, classes=classes, features=features)
#visualizer.show()
