import statistics

import numpy as np

from CoordinatorModel import CoordinatorModel

import gc
#import cupy as cp

class CoordinatorWeightedProb(CoordinatorModel):

    def __init__(self, models, window_size, threshold):
        CoordinatorModel.__init__(self, models, window_size)
        self.threshold = threshold
        self.weights = []

    #def free_gpu_memory(self, preds):
    #    del preds
    #    cp.get_default_memory_pool().free_all_blocks()
    #    gc.collect()

    def set_threshold(self, threshold):
        self.threshold = threshold

    def __calculate_weights__(self):
        f1_models = [model.f1_score_votes if model.f1_score_votes != 0 else 1 for model in self.models]
        self.weights = f1_models / np.sum(f1_models)
        #self.weights = normalized_weights * 5

    def __calculate_weights_2__(self):
        self.weights = []
        f1_models = []
        min = 1.0
        max = 0.0
        for model in self.models:
            f1 = model.recall
            f1_models.append(f1)
            if f1 < min:
                min = f1
            if f1 > max:
                max = f1
        for f1 in f1_models:
            if min == 0 and max == 0:
                weight = 1
            else:
                weight = (f1 - min) / (max - min)
            self.weights.append(weight)

    def set_metrics_old(self, y_real_sample, probs_normal, adv_y_sample = None, probs_adv = None, which_model = None):
        self.weights = np.array(self.weights).reshape(-1, 1)
        weights_normal = self.weights
        recall_votes_adv  = 0.0
        recall_votes_normal = 0.0
        f1_score_adv = 0.0
        f1_score_normal = 0.0
        acc_adv = 0.0
        acc_normal = 0.0
        precision_adv = 0.0
        precision_normal = 0.0        
        # Recall of coordinator adversarial
        if adv_y_sample is not None:
            weights_adversarial = [self.weights[i] for i in which_model]
            probs_adv = np.array(probs_adv)
            weighted_means_adv = np.sum(probs_adv, axis=0) / np.sum(weights_adversarial)
            y_preds_adv = [1 if mean >= self.threshold else 0 for mean in weighted_means_adv]
            recall_votes_adv = self.get_recall_votes(adv_y_sample, y_preds_adv)
            f1_score_adv = self.get_f1score_votes(adv_y_sample, y_preds_adv)
            acc_adv = self.get_acc_votes(adv_y_sample, y_preds_adv)
            precision_adv = self.get_precision_votes(adv_y_sample, y_preds_adv)
            weights_normal = [self.weights[i] for i in list(set([0, 1, 2, 3, 4]) - set(which_model))] 

        # Recall of normal coordinator
        if which_model is None or len(which_model) < 5:
            probs_normal = np.array(probs_normal)   
            weighted_means_normal = np.sum(probs_normal, axis=0) / np.sum(weights_normal)
            y_preds_normal = [1 if mean >= self.threshold else 0 for mean in weighted_means_normal]
            recall_votes_normal = self.get_recall_votes(y_real_sample, y_preds_normal)
            f1_score_normal = self.get_f1score_votes(y_real_sample, y_preds_normal)
            acc_normal = self.get_acc_votes(y_real_sample, y_preds_normal)
            precision_normal = self.get_precision_votes(y_real_sample, y_preds_normal)

        if recall_votes_adv == 0:
            self.set_recall_votes(recall_votes_normal)
            self.set_f1score_votes(f1_score_normal)
            self.set_acc_votes(acc_normal)
            self.set_precision_votes(precision_normal)
        elif recall_votes_normal == 0: 
            self.set_recall_votes(recall_votes_adv)
            self.set_f1score_votes(f1_score_adv)
            self.set_acc_votes(acc_adv)
            self.set_precision_votes(precision_adv)
        else:
            mean_normal_weights = sum(weights_adversarial) / len(weights_adversarial)
            mean_adv_weights = sum(weights_normal) / len(weights_normal)
            mean_recall__votes = (recall_votes_normal*mean_normal_weights  + recall_votes_adv*mean_adv_weights) / (mean_normal_weights + mean_adv_weights)
            self.set_recall_votes(mean_recall__votes)
            mean_f1_score_votes = (f1_score_normal*mean_normal_weights + recall_votes_adv*mean_adv_weights) / (mean_normal_weights + mean_adv_weights)
            self.set_f1score_votes(mean_f1_score_votes)
            mean_acc_votes = (acc_normal*mean_normal_weights + acc_adv*mean_adv_weights) / (mean_normal_weights + mean_adv_weights)
            self.set_acc_votes(mean_acc_votes)
            mean_precision_votes = (precision_normal*mean_normal_weights + precision_adv*mean_adv_weights) / (mean_normal_weights + mean_adv_weights)
            self.set_precision_votes(mean_precision_votes)

    def model_predict_one_votes(self, x_sample, y_labels, adv_x_sample = None, which_model: list = None):
        probs = []
        self.__calculate_weights__()
        print(f"Weights: {self.weights}")
        for i in range(len(self.models)):
            model = self.models[i]
            adv_obj = False
            if adv_x_sample is not None:
                if i in which_model:
                    adv_obj = True
                    prob = model.evaluate_one_proba_votes(adv_x_sample, y_labels, self.threshold)
                    prob = np.array(prob, dtype=np.float32)  # Convert list to NumPy array
                    #prob = prob * self.weights[i] # revisar
                    probs.append(prob)
                    print(f"Adversarial attack performed in model {i} from models {which_model}."+
                          f"Model name: {str(model).replace('votingSystem.ADSystems.', '').split('.')[0]}")  
            if not adv_obj:
                prob = model.evaluate_one_proba_votes(x_sample, y_labels, self.threshold)
                prob = np.array(prob, dtype=np.float32)  # Convert list to NumPy array
                #prob = prob * self.weights[i] # revisar
                probs.append(prob)
            #self.free_gpu_memory(prob)
        
        probs_np = np.array(probs)
        weighted_means = np.average(probs_np, axis=0, weights=self.weights)
        y_preds = [1 if mean >= self.threshold else 0 for mean in weighted_means]
        self.set_metrics(y_labels, y_preds)