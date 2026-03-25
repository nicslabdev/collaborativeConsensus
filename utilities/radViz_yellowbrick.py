
from yellowbrick.features import RadViz
import pandas as pd


data = pd.read_csv("../datasets/chargingStationStates/chargingStationStates_ccs_millon.csv")

features = ['power', 'status', 'idUser', 'duration']
use_pca = True
percentage_test = 0.3
n_components_pca = 2
contamination = 0.01

# Encode strings to float
data['status'] = data['status'].apply(lambda x: 1 if x == 'active' else 0).astype(int)
data['idUser'] = data['idUser'].apply(lambda x: 1 if isinstance(x, str) else 0).astype(int)

# separate data: train and test
x = data.loc[:, features].values
y = data.loc[:, ['anomaly']].values.ravel()

# Standarization


# Specify the features of interest and the classes of the target
classes = ["normal", "anomaly"]

# Instantiate the visualizer
visualizer = RadViz(classes=classes, features=features)
visualizer.fit(x, y)
visualizer.transform(x)
visualizer.show()

