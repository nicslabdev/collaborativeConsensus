import statistics
import copy
import numpy as np
from abc import ABC, abstractmethod
from sklearn.metrics import recall_score, f1_score, accuracy_score
from sklearn.metrics import precision_score
#import shap
#import matplotlib.pyplot as plt
#from lime import lime_tabular # type: ignore

class CoordinatorModel(ABC):

    def __init__(self, models, window_size):
        super().__init__()
        self.models = []
        for model in models:
            self.models.append(copy.deepcopy(model))

        self.f1_score_votes = 0.0
        self.recall_votes = 0.0
        self.acc_votes = 0.0
        self.precision_votes = 0.0

    def reset_coordinator(self):
        for model in self.models:
            model.reset_model()
        self.f1_score_votes = 0.0
        self.recall_votes = 0.0
        self.acc_votes = 0.0
        self.precision_votes = 0.0
#
    #def explain_data(self, model, x_samples, save_path):
    #    if save_path is not None:
    #        if "Random" in str(model):
    #            explainer = shap.TreeExplainer(model.model)  # TreeExplainer for tree-based models (e.g., XGBoost, LightGBM)
    #            shap_values = explainer.shap_values(x_samples)
    #            feature_names = ["connectorType", "durationSession", "durationCharge", "energy", 'cost', 'tariff', "meanPower", 'maxPower']
    #            plt.figure()
    #            shap.summary_plot(shap_values[:, :, 1], x_samples, feature_names=feature_names, show=False)
    #            plt.savefig(save_path.replace("#folder#", "shap").replace("#model#", 
    #                                                                    f"{str(model).split("(")[0].split(".")[0].replace("<", "")}"), bbox_inches="tight", dpi=200)
    #            plt.close()
#
    #            lime_explainer = lime_tabular.LimeTabularExplainer(x_samples, feature_names=feature_names, 
    #                                                            class_names=['Normal', 'Anomalous'], mode="classification")
    #            instance_idx = 5  # Choose a sample index
    #            exp_data = lime_explainer.explain_instance(x_samples[instance_idx], model.model.predict_proba)
    #            fig_lime = exp_data.as_pyplot_figure()
    #            fig_lime.savefig(save_path.replace("#folder#", "lime"), dpi=300, bbox_inches='tight')  # Save as PNG
    
    def model_predict_one_votes(self, x_sample, y_labels, adv_x_sample = None, which_model: list = None):
        pass

    def model_predict_some_votes(self, x_samples, y_labels, adv_x_samples = None, which_model: list = None):
        # Procesar todo del tirón
        if adv_x_samples is not None:
            self.model_predict_one_votes(x_samples, y_labels, adv_x_samples, which_model)
        else:
            self.model_predict_one_votes(x_samples, y_labels)
    
    def calculate_recall_votes(self, true_labels, predictions):
        recall_votes = recall_score(true_labels, predictions)
        self.recall_votes = recall_votes

    def calculate_f1score_votes(self, true_labels, predictions):
        f1_score_votes = f1_score(true_labels, predictions)
        self.f1_score_votes = f1_score_votes

    def calculate_acc_votes(self, true_labels, predictions):
        acc_votes = accuracy_score(true_labels, predictions)
        self.acc_votes = acc_votes
    
    def calculate_precision_votes(self, true_labels, predictions):
        precision_votes = precision_score(true_labels, predictions)
        self.precision_votes = precision_votes
    
    def set_f1score_votes(self, f1score):
        self.f1_score_votes = f1score

    def set_recall_votes(self, recall):
        self.recall_votes = recall

    def set_acc_votes(self, acc):
        self.acc_votes = acc
    
    def set_precision_votes(self, precision):
        self.precision_votes = precision

    def set_metrics(self, y_labels, y_preds):
        self.calculate_recall_votes(y_labels, y_preds)
        self.calculate_f1score_votes(y_labels, y_preds)
        self.calculate_acc_votes(y_labels, y_preds)
        self.calculate_precision_votes(y_labels, y_preds)
    
    def get_logit_votes_models(self):
        logits_models = []
        for model in self.models:
            logits_models.append(model.get_logit_votes())
        return logits_models
    
    def get_entropy_votes_models(self):
        entropy_models = []
        for model in self.models:
            entropy_models.append(model.get_entropy_votes())
        return entropy_models
    
    def get_recall_votes_models(self):
        recall_models = []
        for model in self.models:
            recall_models.append(model.get_recall_votes())
        return recall_models
    
    def get_f1_score_votes_models(self):
        f1_score_models = []
        for model in self.models:
            f1_score_models.append(model.get_f1_score_votes())
        return f1_score_models

    def get_acc_votes_models(self):
        acc_models = []
        for model in self.models:
            acc_models.append(model.get_acc_votes())
        return acc_models

    def get_precision_votes_models(self):
        precision_models = []
        for model in self.models:
            precision_models.append(model.get_precision_votes())
        return precision_models

    #def retrain_models(self):
    #    for model in self.models:
    #        model.retrain_model()