# ANALYSIS 1: EXPERIMENTO 1 
import os, sys, inspect
sys.stdout.reconfigure(line_buffering=True)
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
#parentdir = os.path.dirname(currentdir)
#sys.path.insert(0, parentdir)

from ADSystems.CatboostADS import CatboostADS
import pandas as pd
from functions import encode_variable
from ADSystems.LGBMachineADS import LGBMachineADS
from ADSystems.MultiLayerPerceptronADS import MultiLayerPerceptronADS
from ADSystems.RandomForestADS import RandomForestADS
from ADSystems.XGBoostADS import XGBoostADS
from CoordinatorMax import CoordinatorMax
from CoordinatorProb import CoordinatorProb
from CoordinatorWeightedProb import CoordinatorWeightedProb
import numpy as np
import random 
from itertools import product
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
from scipy import stats
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


files = ['Boulder_Clean_With_Anomalies', 'Dundee_Clean_With_Anomalies', 'Netherlands_Clean_With_Anomalies',
         'PaloAlto_2018_2022_Clean_With_Anomalies', 'Perth_Clean_With_Anomalies', ]


boulderData = pd.read_excel("./data/Boulder_Clean_With_Anomalies.xlsx")
dundeeData = pd.read_excel("./data/Dundee_Clean_With_Anomalies.xlsx")
netherlandsData = pd.read_excel("./data/Netherlands_Clean_With_Anomalies.xlsx")
paloaltoData = pd.read_excel("./data/PaloAlto_2018_2022_Clean_With_Anomalies.xlsx")
perthData = pd.read_excel("./data/Perth_Clean_With_Anomalies.xlsx")


boulderData = boulderData.sort_values(by='startTimestamp', ascending=True)
dundeeData = dundeeData.sort_values(by='startTimestamp', ascending=True)
netherlandsData = netherlandsData.sort_values(by='startTimestamp', ascending=True)
paloaltoData = paloaltoData.sort_values(by='startTimestamp', ascending=True)
perthData = perthData.sort_values(by='startTimestamp', ascending=True)

encode_variable(boulderData, boulderData.columns.get_loc("connectorType"))
encode_variable(dundeeData, dundeeData.columns.get_loc("connectorType"))
encode_variable(netherlandsData, netherlandsData.columns.get_loc("connectorType"))
encode_variable(paloaltoData, paloaltoData.columns.get_loc("connectorType"))
encode_variable(perthData, perthData.columns.get_loc("connectorType"))

features = ["connectorType", "durationSession", "durationCharge", "energy", 'cost', 'tariff', "meanPower", 'maxPower']

x_boulder = boulderData.loc[:, features].values
y_boulder = boulderData.loc[:, ['anomaly']].values.ravel()
x_boulder = scaler.fit_transform(x_boulder)
size = len(x_boulder)
size_init = int(size * 0.8)
x_boulder_init = x_boulder[:size_init]
y_boulder_init = y_boulder[:size_init]
x_boulder_inc = x_boulder[size_init:]
y_boulder_inc = y_boulder[size_init:]
window_boulder_size = 1000
boulderADS = CatboostADS(window_boulder_size, x_boulder_init, y_boulder_init, "BoulderADS_A2E1")

x_dundee = dundeeData.loc[:, features].values
y_dundee = dundeeData.loc[:, ['anomaly']].values.ravel()
x_dundee = scaler.fit_transform(x_dundee)
size = len(x_dundee)
size_init = int(size * 0.8)
x_dundee_init = x_dundee[:size_init]
y_dundee_init = y_dundee[:size_init]
x_dundee_inc = x_dundee[size_init:]
y_dundee_inc = y_dundee[size_init:]
window_dundee_size = 1000
dundeeADS = LGBMachineADS(window_dundee_size, x_dundee_init, y_dundee_init, "DundeeADS_A2E1")

x_netherlands = netherlandsData.loc[:, features].values
y_netherlands = netherlandsData.loc[:, ['anomaly']].values.ravel()
x_netherlands = scaler.fit_transform(x_netherlands)
size = len(x_netherlands)
size_init = int(size * 0.8)
x_netherlands_init = x_netherlands[:size_init]
y_netherlands_init = y_netherlands[:size_init]
x_netherlands_inc = x_netherlands[size_init:]
y_netherlands_inc = y_netherlands[size_init:]
window_netherlands_size = 1000
netherlandsADS = MultiLayerPerceptronADS(window_netherlands_size, x_netherlands_init, y_netherlands_init,
                                         "NetherlandsADS_A2E1")

x_paloalto = paloaltoData.loc[:, features].values
y_paloalto = paloaltoData.loc[:, ['anomaly']].values.ravel()
x_paloalto = scaler.fit_transform(x_paloalto)
size = len(x_paloalto)
size_init = int(size * 0.8)
x_paloalto_init = x_paloalto[:size_init]
y_paloalto_init = y_paloalto[:size_init]
x_paloalto_inc = x_paloalto[size_init:]
y_paloalto_inc = y_paloalto[size_init:]
window_paloalto_size = 1000
paloaltoADS = RandomForestADS(window_paloalto_size, x_paloalto_init, y_paloalto_init, "PaloAltoADS_A2E1")

x_perth = perthData.loc[:, features].values
y_perth = perthData.loc[:, ['anomaly']].values.ravel()
x_perth = scaler.fit_transform(x_perth)
size = len(x_perth)
size_init = int(size * 0.8)
x_perth_init = x_perth[:size_init]
y_perth_init = y_perth[:size_init]
x_perth_inc = x_perth[size_init:]
y_perth_inc = y_perth[size_init:]
window_perth_size = 1000
perthADS = XGBoostADS(window_perth_size, x_perth_init, y_perth_init, "PerthADS_A2E1")

adsModels = [boulderADS, dundeeADS, netherlandsADS, paloaltoADS, perthADS]
# equivalencias: 1: Catboost, 2: LGB, 3: MLP, 4: RF, 5: XGBoost
window_size1 = 1000

coordinator1 = CoordinatorProb(adsModels, window_size1, 0.45)
coordinator3 = CoordinatorWeightedProb(adsModels, window_size1, 0.45)
coordinator5 = CoordinatorMax(adsModels, window_size1)

x_inc = np.concatenate((x_boulder_inc, x_netherlands_inc), 0)
x_inc = np.concatenate((x_inc, x_dundee_inc), 0)
x_inc = np.concatenate((x_inc, x_paloalto_inc), 0)
x_inc = np.concatenate((x_inc, x_perth_inc), 0)

y_inc = np.concatenate((y_boulder_inc, y_netherlands_inc), 0)
y_inc = np.concatenate((y_inc, y_dundee_inc), 0)
y_inc = np.concatenate((y_inc, y_paloalto_inc), 0)
y_inc = np.concatenate((y_inc, y_perth_inc), 0)

df = pd.DataFrame(x_inc, columns=features)
df["anomaly"] = y_inc
#df = df.sample(frac=1).reset_index(drop=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

x_inc = df.loc[:, features].values
y_inc = df.loc[:, ['anomaly']].values.ravel()

def trim_array_to_equal_parts(arr, num_parts):
    # Calculate the maximum number of elements that can be equally divided
    total_elements = len(arr)
    divisible_length = total_elements - (total_elements % num_parts)
    
    # Trim the array to make it divisible by num_parts
    return arr[:divisible_length]

num_splits = 50
x_splits_noatt = np.split(trim_array_to_equal_parts(x_inc, num_splits), num_splits) 
y_splits_noatt = np.split(trim_array_to_equal_parts(y_inc, num_splits), num_splits)
print("Shape of no attack data splits.")
print(x_splits_noatt[num_splits - 1].shape)
print(y_splits_noatt[num_splits - 1].shape)


def read_adv_data(advPath):
    features = ['connectortype', 'durationcharge' , 'durationsession' , 'energy' , 'tariff' ,
                'cost' , 'meanpower', 'maxpower']
    adversarial_data = pd.read_csv(advPath)
    adversarial_x_data = adversarial_data.loc[:, features].values
    adversarial_y_data = adversarial_data.loc[:, ['anomaly']].values.ravel()

    return adversarial_x_data, adversarial_y_data 

def generate_random_attack_models(attack: str, eps):
    if attack == "random":
        which_models = [[random.randint(0, 4)] for _ in range(50)]
    elif attack == "weak":
        which_models = [[3] for _ in range(50)]
    elif attack == "strong":
        which_models = [[2] for _ in range(50)]
    elif attack == "weak_strong":
        which_models = [[2, 3] for _ in range(50)]
    elif attack == "all":
        which_models = [[0, 1, 2, 3, 4] for _ in range(50)]
    elif attack == "three_random":
        which_model = [2, 3, 0] #random.sample(range(0, 4), 3)
        which_models = [which_model for _ in range(50)]
    elif attack == "No attack":
        which_models = [[]]
        pass
    else:
        raise Exception("Error, no attack defined.")
    return which_models

def cargar_muestras_adversarias(eps, x_dundee_inc, y_dundee_inc, num_splits, x_boulder_inc, y_boulder_inc,  x_netherlands_inc, y_netherlands_inc,
                                x_paloalto_inc, y_paloalto_inc, x_perth_inc, y_perth_inc, features: list, y_inc_original, name):
    # Cargar muestras adversarias
    adv_path = f'/mnt/AI-DATA/imanb/advDetection/data/aexamples/{name}/mifgsm/test/adversarialexamples_eps{str(eps)}.csv'
    adversarial_x_data, adversarial_y_data = read_adv_data(adv_path)
    x_test_adversarial = np.array(adversarial_x_data, dtype=np.float32) # Datos anomalos adversarios
    y_test_adversarial = np.array(adversarial_y_data, dtype=np.float32) # hay que mezclar con datos normales

    if 'Perth' in name:
        anomalous_indices = np.where(y_perth_inc == 1)[0]
        X_modified, Y_modified  = x_perth_inc.copy(), y_perth_inc.copy()
        X_modified[anomalous_indices] = x_test_adversarial
        Y_modified[anomalous_indices] = y_test_adversarial

        x_inc = np.concatenate((x_boulder_inc, x_netherlands_inc), 0)
        x_inc = np.concatenate((x_inc, x_dundee_inc), 0)
        x_inc = np.concatenate((x_inc, x_paloalto_inc), 0)
        x_inc = np.concatenate((x_inc, X_modified), 0) # x_perth_inc

        y_inc = np.concatenate((y_boulder_inc, y_netherlands_inc), 0)
        y_inc = np.concatenate((y_inc, y_dundee_inc), 0)
        y_inc = np.concatenate((y_inc, y_paloalto_inc), 0)
        y_inc = np.concatenate((y_inc, Y_modified), 0) #y_perth_inc
    elif 'Boulder' in name:
        anomalous_indices = np.where(y_boulder_inc == 1)[0]
        X_modified, Y_modified  = x_boulder_inc.copy(), y_boulder_inc.copy()
        X_modified[anomalous_indices] = x_test_adversarial
        Y_modified[anomalous_indices] = y_test_adversarial

        x_inc = np.concatenate((X_modified, x_netherlands_inc), 0)
        x_inc = np.concatenate((x_inc, x_dundee_inc), 0)
        x_inc = np.concatenate((x_inc, x_paloalto_inc), 0)
        x_inc = np.concatenate((x_inc, x_perth_inc), 0) # x_perth_inc

        y_inc = np.concatenate((Y_modified, y_netherlands_inc), 0)
        y_inc = np.concatenate((y_inc, y_dundee_inc), 0)
        y_inc = np.concatenate((y_inc, y_paloalto_inc), 0)
        y_inc = np.concatenate((y_inc, y_perth_inc), 0) #y_perth_inc
    elif 'Dundee' in name:
        anomalous_indices = np.where(y_dundee_inc == 1)[0]
        X_modified, Y_modified  = x_dundee_inc.copy(), y_dundee_inc.copy()
        X_modified[anomalous_indices] = x_test_adversarial
        Y_modified[anomalous_indices] = y_test_adversarial

        x_inc = np.concatenate((x_boulder_inc, x_netherlands_inc), 0)
        x_inc = np.concatenate((x_inc, X_modified), 0)
        x_inc = np.concatenate((x_inc, x_paloalto_inc), 0)
        x_inc = np.concatenate((x_inc, x_perth_inc), 0) # x_perth_inc

        y_inc = np.concatenate((y_boulder_inc, y_netherlands_inc), 0)
        y_inc = np.concatenate((y_inc, Y_modified), 0)
        y_inc = np.concatenate((y_inc, y_paloalto_inc), 0)
        y_inc = np.concatenate((y_inc, y_perth_inc), 0) #y_perth_inc
    elif 'Netherland' in name:
        anomalous_indices = np.where(y_netherlands_inc == 1)[0]
        X_modified, Y_modified  = x_netherlands_inc.copy(), y_netherlands_inc.copy()
        X_modified[anomalous_indices] = x_test_adversarial
        Y_modified[anomalous_indices] = y_test_adversarial

        x_inc = np.concatenate((x_boulder_inc, X_modified), 0)
        x_inc = np.concatenate((x_inc, x_dundee_inc), 0)
        x_inc = np.concatenate((x_inc, x_paloalto_inc), 0)
        x_inc = np.concatenate((x_inc, x_perth_inc), 0) # x_perth_inc

        y_inc = np.concatenate((y_boulder_inc, Y_modified), 0)
        y_inc = np.concatenate((y_inc, y_dundee_inc), 0)
        y_inc = np.concatenate((y_inc, y_paloalto_inc), 0)
        y_inc = np.concatenate((y_inc, y_perth_inc), 0) #y_perth_inc
    elif 'Palo' in name:
        anomalous_indices = np.where(y_paloalto_inc== 1)[0]
        X_modified, Y_modified  = x_paloalto_inc.copy(), y_paloalto_inc.copy()
        X_modified[anomalous_indices] = x_test_adversarial
        Y_modified[anomalous_indices] = y_test_adversarial

        x_inc = np.concatenate((x_boulder_inc, x_netherlands_inc), 0)
        x_inc = np.concatenate((x_inc, x_dundee_inc), 0)
        x_inc = np.concatenate((x_inc, X_modified), 0)
        x_inc = np.concatenate((x_inc, x_perth_inc), 0) # x_perth_inc

        y_inc = np.concatenate((y_boulder_inc, y_netherlands_inc), 0)
        y_inc = np.concatenate((y_inc, y_dundee_inc), 0)
        y_inc = np.concatenate((y_inc, Y_modified), 0)
        y_inc = np.concatenate((y_inc, y_perth_inc), 0) #y_perth_inc


    df = pd.DataFrame(x_inc, columns=features)
    df["anomaly"] = y_inc
    #df = df.sample(frac=1).reset_index(drop=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    x_inc = df.loc[:, features].values
    y_inc = df.loc[:, ['anomaly']].values.ravel()
    
    if np.array_equal(y_inc, y_inc_original):
        print("Arrays are the same")

    x_splits_adv = np.split(trim_array_to_equal_parts(x_inc, num_splits), num_splits) 
    y_splits_adv = np.split(trim_array_to_equal_parts(y_inc, num_splits), num_splits)
    print("Shape of adv data splits.")
    print(x_splits_adv[num_splits - 1].shape)
    print(y_splits_adv[num_splits - 1].shape)
    return x_splits_adv, y_splits_adv

def set_y_limits(metric):
    if metric == "Accuracy" or metric == "Recall" or metric == "F1 Score":
            y_min = 0.4 # Store min values for each metric
            y_max = 1.05  # Store max values for each metric
    else:
        y_min = -0.2
        y_max = 1.0
    return y_min, y_max

def plot(title, coordinator_name, data_coordinators, data_models, plot_file_name, metric, lower_ci = 0, upper_ci = 0):
    data_coordinators = [float(x) if isinstance(x, np.ndarray) else x for x in data_coordinators]
    plt.figure(figsize=(10, 4))  # Create a new figure for each coordinator
    
    # Convert coordinator's data to a DataFrame
    df_coordinator = pd.DataFrame({
        "Step": range(len(data_coordinators)),
        metric: data_coordinators,
        "Type": coordinator_name
    })

    # Convert model data to a DataFrame and Melt df_models to long format for Seaborn plotting
    df_models = pd.DataFrame(data_models, columns=["CB", "LGBM", "MLP", "RF", "XGB"])
    df_models["Step"] = df_models.index  # Add a time step index
    df_models_long = df_models.melt(id_vars=["Step"], var_name="Model", value_name=metric)
    
    y_min, y_max = set_y_limits(metric)
    # Plot the coordinator data
    sns.lineplot(data=df_coordinator, x="Step", y=metric, label=f"Coordinator {coordinator_name}", color="black", linewidth=2, marker="o", rasterized=True)

    # Plot model predictions
    sns.lineplot(data=df_models_long, x="Step", y=metric, hue="Model", palette="rocket", linestyle="dashed", rasterized=True)

    plt.ylim(y_min, y_max)

    if 'Recall' in metric:
        plt.axhline(
            y=lower_ci,
            color="gray",
            linestyle="--",
            linewidth=2,
            label=f"Lower 95% CI ({lower_ci:.4f})"
        )
        plt.axhline(
            y=upper_ci,
            color="gray",
            linestyle="--",
            linewidth=2,
            label=f"Upper 95% CI ({upper_ci:.4f})"
        )

    # Add labels and title
    plt.xlabel("Split")
    plt.ylabel(metric)
    plt.title(f"{coordinator_name} - {title}")
    plt.legend(title="Legend")
    plt.xticks(np.arange(0, 51, 5))
    #plt.savefig(plot_file_name, dpi=300, bbox_inches='tight')  # Save as a high-quality PNG
    plt.savefig(plot_file_name, format="pdf", dpi=200, bbox_inches='tight')
    plt.clf()
   
epsilon = [0.0, 0.021, 0.090, 0.210]

# Preparar tabla de resultados
metrics = ["Accuracy", "F1 Score", "Recall", "EIR"]
model_names = ["COORDINATOR", "CATBOOST", "LGBM", "MLP", "RF", "XGB"]
coordinators = ["Mean", "WeightedMean", "Mode"]  
attacks = ["No attack", "all"]
multi_index = pd.MultiIndex.from_tuples(
    list(product(attacks, coordinators, model_names, metrics)), 
    names=["Attack", "Coordinator", "Model", "Metric"]
)
df_results = pd.DataFrame(index=multi_index, columns=epsilon)
print(f"Starting attacks")
for file in files:
    print(f'Processing file {file}.')
    for eps in epsilon:
        print(f"** Epsilon value: {eps} **")
        if eps == 0:
            attacks = ["No attack"]
            coord_1_recall, coord_3_recall, coord_5_recall   = [], [], [] # no attack eps 0
            models_coord1_recall, models_coord3_recall, models_coord5_recall = [], [], [] # no attack eps 0
        else:
            attacks = ["all"] #random #,
        for att_idx, attack in enumerate(attacks):
            # Inicializamos modelos de ataque
            print(f"----- Current attack: {attack} **")
            if eps > 0 : #Attack
                x_splits_adv, y_splits_adv =  cargar_muestras_adversarias(eps, x_dundee_inc, y_dundee_inc, num_splits, x_boulder_inc,
                                                            y_boulder_inc,  x_netherlands_inc, y_netherlands_inc,
                                    x_paloalto_inc, y_paloalto_inc, x_perth_inc, y_perth_inc, features, y_inc, file) 
                # Cargar muestras adversarias
                which_models = generate_random_attack_models(attack, eps)

            # Inicializar listas métricas para este ataque (f1, recall, accuracy)
            coordinator1_f1_votes, models_coord1_f1_votes = [], []
            coordinator1_recall_votes, models_coord1_recall_votes = [], []
            coordinator1_eir_votes, models_coord1_eir_votes = [], []
            coordinator1_acc_votes, models_coord1_acc_votes = [], []
            coordinator3_f1_votes, models_coord3_f1_votes = [], []
            coordinator3_recall_votes, models_coord3_recall_votes = [], []
            coordinator3_eir_votes, models_coord3_eir_votes = [], []
            coordinator3_acc_votes, models_coord3_acc_votes = [], []
            coordinator5_f1_votes, models_coord5_f1_votes = [], []
            coordinator5_recall_votes, models_coord5_recall_votes = [], []
            coordinator5_eir_votes, models_coord5_eir_votes = [], []
            coordinator5_acc_votes, models_coord5_acc_votes = [], []

            coordinator5_precision_votes, models_coord5_precision_votes = [], []

            print(f"Staring 50 splits of attack {attack}")
            # Comenzamos bucle: 50 splits
            for i in range(0, num_splits):
                if eps == 0: # No attack
                    x_samples = x_splits_noatt[i] # Datos normales y anómalos
                    y_samples = y_splits_noatt[i]
                    
                    coordinator1.model_predict_some_votes(x_samples, y_samples)
                    coordinator3.model_predict_some_votes(x_samples, y_samples)
                    coordinator5.model_predict_some_votes(x_samples, y_samples)
                    coord_1_recall.append(coordinator1.recall_votes)
                    coord_3_recall.append(coordinator3.recall_votes)
                    coord_5_recall.append(coordinator5.recall_votes)
                    models_coord1_recall.append(coordinator1.get_recall_votes_models())
                    models_coord3_recall.append(coordinator3.get_recall_votes_models())
                    models_coord5_recall.append(coordinator5.get_recall_votes_models())

                    coordinator1_eir_votes.append(0.0)
                    models_coord1_eir_votes.append([0, 0, 0, 0, 0])
                    coordinator3_eir_votes.append(0.0)
                    models_coord3_eir_votes.append([0, 0, 0, 0, 0])
                    coordinator5_eir_votes.append(0.0)
                    models_coord5_eir_votes.append([0, 0, 0, 0, 0])
                else: # Attack
                    x_samples = x_splits_noatt[i] # Datos normales y anómalos
                    y_samples = y_splits_noatt[i]
                    adv_x_samples = x_splits_adv[i] # Datos normales y adversarios
                    adv_y_samples = y_splits_adv[i]

                    which_model = which_models[i]
                    print(f"Target models: {which_model}")
                    coordinator1.model_predict_some_votes(x_samples, y_samples, adv_x_samples, which_model)
                    coordinator3.model_predict_some_votes(x_samples, y_samples, adv_x_samples, which_model)
                    coordinator5.model_predict_some_votes(x_samples, y_samples, adv_x_samples, which_model)

                    coordinator1_eir_votes.append(round(1 - (float(coordinator1.recall_votes) / max(coord_1_recall[i], 1e-6)),2))
                    models_coord1_eir_votes.append([round(1 - (now / max(orig, 1e-6)), 2) for now, orig in zip(coordinator1.get_recall_votes_models(), models_coord1_recall[i])])
                    coordinator3_eir_votes.append(round(1 - (float(coordinator3.recall_votes) / max(coord_3_recall[i], 1e-6)),2))
                    models_coord3_eir_votes.append([round(1 - (now / max(orig, 1e-6)), 2)  for now, orig in zip(coordinator3.get_recall_votes_models(), models_coord3_recall[i])])
                    coordinator5_eir_votes.append(round(1 - (float(coordinator5.recall_votes) / max(coord_5_recall[i], 1e-6)),2))
                    models_coord5_eir_votes.append([round(1 - (now / max(orig, 1e-6)), 2)  for now, orig in zip(coordinator5.get_recall_votes_models(), models_coord5_recall[i])])

                coordinator1_f1_votes.append(coordinator1.f1_score_votes) 
                models_coord1_f1_votes.append(coordinator1.get_f1_score_votes_models()) 
                coordinator1_recall_votes.append(coordinator1.recall_votes) 
                models_coord1_recall_votes.append(coordinator1.get_recall_votes_models()) 
                coordinator1_acc_votes.append(coordinator1.acc_votes)  
                models_coord1_acc_votes.append(coordinator1.get_acc_votes_models()) 

                coordinator3_f1_votes.append(coordinator3.f1_score_votes) 
                models_coord3_f1_votes.append(coordinator3.get_f1_score_votes_models())
                coordinator3_recall_votes.append(coordinator3.recall_votes) 
                models_coord3_recall_votes.append(coordinator3.get_recall_votes_models()) 
                coordinator3_acc_votes.append(coordinator3.acc_votes)  
                models_coord3_acc_votes.append(coordinator3.get_acc_votes_models())

                coordinator5_f1_votes.append(coordinator5.f1_score_votes) 
                models_coord5_f1_votes.append(coordinator5.get_f1_score_votes_models())
                coordinator5_recall_votes.append(coordinator5.recall_votes) 
                models_coord5_recall_votes.append(coordinator5.get_recall_votes_models()) 
                coordinator5_acc_votes.append(coordinator5.acc_votes)  
                models_coord5_acc_votes.append(coordinator5.get_acc_votes_models())

                coordinator5_precision_votes.append(coordinator5.precision_votes)
                models_coord5_precision_votes.append(coordinator5.get_precision_votes_models())
                print(f"----> Split {str(i)} done")

            df_results.loc[(attack,"Mean", "COORDINATOR", "F1 Score"), eps] = round(statistics.mean(map(float, coordinator1_f1_votes)) , 2)
            df_results.loc[(attack,"Mean", "COORDINATOR", "Recall"), eps] = round(statistics.mean(map(float, coordinator1_recall_votes)) , 2)
            df_results.loc[(attack,"Mean", "COORDINATOR", "Accuracy"), eps] = round(statistics.mean(map(float,coordinator1_acc_votes)) , 2)

            
            df_results.loc[(attack,"WeightedMean", "COORDINATOR", "F1 Score"), eps] = round(statistics.mean(map(float,coordinator3_f1_votes)) , 2)
            df_results.loc[(attack,"WeightedMean", "COORDINATOR", "Recall"), eps] = round(statistics.mean(map(float,coordinator3_recall_votes)) , 2)
            df_results.loc[(attack,"WeightedMean", "COORDINATOR", "Accuracy"), eps] = round(statistics.mean(map(float,coordinator3_acc_votes)) , 2)
    
            df_results.loc[(attack,"Mode", "COORDINATOR", "F1 Score"), eps] = round(statistics.mean(map(float,coordinator5_f1_votes)) , 2)
            df_results.loc[(attack,"Mode", "COORDINATOR", "Recall"), eps] = round(statistics.mean(map(float,coordinator5_recall_votes)) , 2)  
            df_results.loc[(attack,"Mode", "COORDINATOR", "Accuracy"), eps] = round(statistics.mean(map(float,coordinator5_acc_votes)) , 2)
        
            # EIR coordinators
            if eps == 0:
                df_results.loc[(attack, "Mean", "COORDINATOR", "EIR"), eps] = "-"
                df_results.loc[(attack, "WeightedMean", "COORDINATOR", "EIR"), eps] = "-"
                df_results.loc[(attack, "Mode", "COORDINATOR", "EIR"), eps] = "-"
            else:
                df_results.loc[(attack, "Mean", "COORDINATOR", "EIR"), eps] = round(statistics.mean(map(float,coordinator1_eir_votes)),2)
                df_results.loc[(attack, "WeightedMean", "COORDINATOR", "EIR"), eps] = round(statistics.mean(map(float,coordinator3_eir_votes)),2)
                df_results.loc[(attack, "Mode", "COORDINATOR", "EIR"), eps] = round(statistics.mean(map(float, coordinator5_eir_votes)),2)
        # Coord 1: ADS metrics
            c1_f1_mean_models = np.round(np.array(models_coord1_f1_votes).mean(axis=0), 2).tolist()
            for idx, mean_c1_f1_model in enumerate(c1_f1_mean_models):
                model_name = model_names[1:][idx]
                df_results.loc[(attack,"Mean", model_name, "F1 Score"), eps] = mean_c1_f1_model

            c1_recall_mean_models = np.round(np.array(models_coord1_recall_votes).mean(axis=0),2).tolist()
            for idx, mean_c1_recall_model in enumerate(c1_recall_mean_models):
                model_name = model_names[1:][idx]
                df_results.loc[(attack,"Mean", model_name, "Recall"), eps] = mean_c1_recall_model
                        # EIR coordinators
                if eps == 0:
                    df_results.loc[(attack, "Mean", model_name, "EIR"), eps] = "-"
                else:
                    df_results.loc[(attack, "Mean", model_name, "EIR"), eps] = 1 - (mean_c1_recall_model / max(df_results.loc[("No attack", "Mean", model_name, "Recall"), 0.0], 1e-6))
        

            c1_acc_mean_models = np.round(np.array(models_coord1_acc_votes).mean(axis=0),2).tolist()
            for idx, mean_c1_acc_model in enumerate(c1_acc_mean_models):
                model_name = model_names[1:][idx]
                df_results.loc[(attack,"Mean", model_name, "Accuracy"), eps] = mean_c1_acc_model

            # Coord 3: ADS metrics
            c3_f1_mean_models = np.round(np.array(models_coord3_f1_votes).mean(axis=0), 2).tolist()
            for idx, mean_c3_f1_model in enumerate(c3_f1_mean_models):
                model_name = model_names[1:][idx]
                df_results.loc[(attack,"WeightedMean", model_name, "F1 Score"), eps] = mean_c3_f1_model
            
            c3_recall_mean_models = np.round(np.array(models_coord3_recall_votes).mean(axis=0),2).tolist()
            for idx, mean_c3_recall_model in enumerate(c3_recall_mean_models):
                model_name = model_names[1:][idx]
                df_results.loc[(attack,"WeightedMean", model_name, "Recall"), eps] = mean_c3_recall_model
                if eps == 0:
                    df_results.loc[(attack, "WeightedMean", model_name, "EIR"), eps] = "-"
                else:
                    df_results.loc[(attack, "WeightedMean", model_name, "EIR"), eps] = 1 - (mean_c3_recall_model / max(df_results.loc[("No attack", "WeightedMean", model_name, "Recall"), 0.0], 1e-6))
            
            c3_acc_mean_models = np.round(np.array(models_coord3_acc_votes).mean(axis=0),2).tolist()
            for idx, mean_c3_acc_model in enumerate(c3_acc_mean_models):
                model_name = model_names[1:][idx]
                df_results.loc[(attack,"WeightedMean", model_name, "Accuracy"), eps] = mean_c3_acc_model
            
            #Coord 5: ADS metrics
            c5_f1_mean_models = np.round(np.array(models_coord5_f1_votes).mean(axis=0), 2).tolist()
            for idx, mean_c5_f1_model in enumerate(c5_f1_mean_models):
                model_name = model_names[1:][idx]
                df_results.loc[(attack,"Mode", model_name, "F1 Score"), eps] = mean_c5_f1_model

            c5_recall_mean_models = np.round(np.array(models_coord5_recall_votes).mean(axis=0),2).tolist()
            for idx, mean_c5_recall_model in enumerate(c5_recall_mean_models):
                model_name = model_names[1:][idx]
                df_results.loc[(attack,"Mode", model_name, "Recall"), eps] = mean_c5_recall_model
                if eps == 0:
                    df_results.loc[(attack, "Mode", model_name, "EIR"), eps] = "-"
                else:
                    df_results.loc[(attack, "Mode", model_name, "EIR"), eps] = 1 - (mean_c5_recall_model / max(df_results.loc[("No attack", "Mode", model_name, "Recall"), 0.0], 1e-6))

            c5_acc_mean_models = np.round(np.array(models_coord5_acc_votes).mean(axis=0),2).tolist()
            for idx, mean_c5_acc_model in enumerate(c5_acc_mean_models):
                model_name = model_names[1:][idx]
                df_results.loc[(attack,"Mode", model_name, "Accuracy"), eps] = mean_c5_acc_model

            recall_normal = np.array(coord_3_recall)
            recall_attack = np.array(coordinator3_recall_votes)
            diff = recall_attack - recall_normal
            mean_diff = np.mean(diff)
            ci = stats.t.interval(0.95, len(diff)-1, loc=mean_diff, scale=stats.sem(diff))
            t_stat, p_two_tailed = stats.ttest_rel(recall_attack, recall_normal)

            # Convertir a prueba unilateral: H1: mu_attack < mu_normal
            if t_stat < 0:
                p_one_tailed = p_two_tailed / 2
            else:
                p_one_tailed = 1 - p_two_tailed / 2
            p_value = p_one_tailed
            print("-----------------------------------------------------------------")
            print(f"Diferencia media: {mean_diff:.4f}")
            print(f"IC 95%: [{ci[0]:.4f}, {ci[1]:.4f}]")
            print(f"t = {t_stat:.3f}, p = {p_value:.6f}")
            print("-----------------------------------------------------------------")
            mean_normal = recall_normal.mean(axis=0)
            sem_normal = stats.sem(recall_normal, axis=0)  # error estándar
            ci95 = 1.96 * sem_normal                      # 95% usando z (aprox.)
            lower_normal = mean_normal - ci95
            upper_normal = mean_normal + ci95
            print(f"upper ci: {upper_normal} lower ci: {lower_normal}")
            print("-----------------------------------------------------------------")
            df_results_reset = df_results.reset_index()  # Convert MultiIndex to columns
            graph_folder = f"./graphs/A2E1/A2E1_wb_{file.split('_')[0]}"
            os.makedirs(graph_folder, exist_ok=True)
            df_results_reset.to_excel(f"{graph_folder}/0_summary_attack_results.xlsx", index=False)

            
            plot(f"F1-score {attack} eps: {eps}", "Mean", coordinator1_f1_votes, models_coord1_f1_votes, f"./{graph_folder}/A2E1_Attack[{att_idx}]_{attack}_eps_0.{str(eps).split('.')[1]}_f1_plot_coord1.pdf", "F1 Score")
            plot(f"Recall {attack} eps: {eps}", "Mean", coordinator1_recall_votes, models_coord1_recall_votes, f"./{graph_folder}/A2E1_Attack[{att_idx}]_{attack}_eps_0.{str(eps).split('.')[1]}_recall_plot_coord1.pdf", "Recall", lower_normal, upper_normal)
            plot(f"Accuracy {attack} eps: {eps}", "Mean", coordinator1_acc_votes, models_coord1_acc_votes, f"./{graph_folder}/A2E1_Attack[{att_idx}]_{attack}_eps_0.{str(eps).split('.')[1]}_acc_plot_coord1.pdf", "Accuracy")

            plot(f"F1-score {attack} eps: {eps}", "WeightedMean", coordinator3_f1_votes, models_coord3_f1_votes, f"./{graph_folder}/A2E1_Attack[{att_idx}]_{attack}_eps_0.{str(eps).split('.')[1]}_f1_plot_coord3.pdf", "F1 Score")
            plot(f"Recall {attack} eps: {eps}", "WeightedMean", coordinator3_recall_votes, models_coord3_recall_votes, f"./{graph_folder}/A2E1_Attack[{att_idx}]_{attack}_eps_0.{str(eps).split('.')[1]}_recall_plot_coord3.pdf", "Recall", lower_normal, upper_normal)
            plot(f"Accuracy {attack} eps: {eps}", "WeightedMean", coordinator3_acc_votes, models_coord3_acc_votes, f"./{graph_folder}/A2E1_Attack[{att_idx}]_{attack}_eps_0.{str(eps).split('.')[1]}_acc_plot_coord3.pdf", "Accuracy")

            plot(f"F1-score {attack} eps: {eps}", "Mode", coordinator5_f1_votes, models_coord5_f1_votes, f"./{graph_folder}/A2E1_Attack[{att_idx}]_{attack}_eps_0.{str(eps).split('.')[1]}_f1_plot_coord5.pdf", "F1 Score")
            plot(f"Recall {attack} eps: {eps}", "Mode", coordinator5_recall_votes, models_coord5_recall_votes, f"./{graph_folder}/A2E1_Attack[{att_idx}]_{attack}_eps_0.{str(eps).split('.')[1]}_recall_plot_coord5.pdf", "Recall", lower_normal, upper_normal)
            plot(f"Accuracy {attack} eps: {eps}", "Mode", coordinator5_acc_votes, models_coord5_acc_votes, f"./{graph_folder}/A2E1_Attack[{att_idx}]_{attack}_eps_0.{str(eps).split('.')[1]}_acc_plot_coord5.pdf", "Accuracy")
            #plot(f"Precision {attack} eps: {eps}", "Mode", coordinator5_precision_votes, models_coord5_precision_votes, f"./{graph_folder}/A2E1_Attack[{att_idx}]_{attack}_eps_0.{str(eps).split('.')[1]}_precision_plot_coord5.pdf", "Precision")

            plot(f"EIR {attack} eps: {eps}", "Mean", coordinator1_eir_votes, models_coord1_eir_votes, f"./{graph_folder}/A2E1_Attack[{att_idx}]_{attack}_eps_0.{str(eps).split('.')[1]}_eir_plot_coord1.pdf", "EIR")
            plot(f"EIR {attack} eps: {eps}", "WeightedMean", coordinator3_eir_votes, models_coord3_eir_votes, f"./{graph_folder}/A2E1_Attack[{att_idx}]_{attack}_eps_0.{str(eps).split('.')[1]}_eir_plot_coord3.pdf", "EIR")
            plot(f"EIR {attack} eps: {eps}", "Mode", coordinator5_eir_votes, models_coord5_eir_votes, f"./{graph_folder}/A2E1_Attack[{att_idx}]_{attack}_eps_0.{str(eps).split('.')[1]}_eir_plot_coord5.pdf", "EIR")      

print("A2E1 finished successfully!!")