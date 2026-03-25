
import sys, os, inspect
from joblib import load

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir)

if __name__ == "__main__":
    #model = load("anomaliesDetection/models/sessionModel/acnData/algorithms/autoML/acndata_autoML_allFeatures_AML.joblib")
    model = load("anomaliesDetection/models/sessionModel/acnData/algorithms/autoML/acndata_autoML_allFeatures_AML.joblib")
    #print(model.sprint_statistics())
    print(model.show_models())
