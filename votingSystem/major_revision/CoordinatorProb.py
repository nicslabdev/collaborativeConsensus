import statistics

import numpy as np

from CoordinatorModel import CoordinatorModel
import gc
#import cupy as cp

class CoordinatorProb(CoordinatorModel):

    def __init__(self, models, window_size, threshold):
        CoordinatorModel.__init__(self, models, window_size)
        self.threshold = threshold

    #def free_gpu_memory(self, preds):
    #    del preds
    #    cp.get_default_memory_pool().free_all_blocks()
    #    gc.collect()

    def set_threshold(self, threshold):
        self.threshold = threshold
    
    def set_metrics_old(self, y_real_sample, probs_normal, adv_y_sample = None, probs_adv = None, which_model = None):
        # Metrics of coordinator 
        recall_votes_adv  = 0.0
        recall_votes_normal = 0.0
        f1_score_adv = 0.0
        f1_score_normal = 0.0
        acc_adv = 0.0
        acc_normal = 0.0
        precision_adv = 0.0
        precision_normal = 0.0
        if adv_y_sample is not None:
            adv_probs_np = np.array(probs_adv)
            adv_means_prob = np.mean(adv_probs_np, axis=0) # calcula las medias de las columnas
            y_preds_adv = [1 if mean >= self.threshold else 0 for mean in adv_means_prob]
            recall_votes_adv = self.get_recall_votes(adv_y_sample, y_preds_adv)
            f1_score_adv = self.get_f1score_votes(adv_y_sample, y_preds_adv)
            acc_adv = self.get_acc_votes(adv_y_sample, y_preds_adv)
            precision_adv = self.get_precision_votes(adv_y_sample, y_preds_adv)
        if which_model is None or len(which_model) < 5:
            # Recall of coordinator - normal
            normal_probs_np = np.array(probs_normal)     
            normal_means_prob = np.mean(normal_probs_np, axis=0) 
            y_preds_normal = [1 if mean >= self.threshold else 0 for mean in normal_means_prob]
            recall_votes_normal = self.get_recall_votes(y_real_sample, y_preds_normal)
            f1_score_normal = self.get_f1score_votes(y_real_sample, y_preds_normal)
            acc_normal = self.get_acc_votes(y_real_sample, y_preds_normal)
            precision_normal = self.get_precision_votes(y_real_sample, y_preds_normal)
        if recall_votes_adv == 0:
            self.set_recall_votes(recall_votes_normal)
            self.set_f1score_votes(f1_score_normal)
            self.set_acc_votes(acc_normal)
            self.set_precision_votes(precision_normal)
            print(recall_votes_normal)
            print(acc_normal)
        elif recall_votes_normal == 0: 
            self.set_recall_votes(recall_votes_adv)
            self.set_f1score_votes(f1_score_adv)
            self.set_acc_votes(acc_adv)
            self.set_precision_votes(precision_adv)
        else:
            mean_recall_votes = (recall_votes_normal + recall_votes_adv) / 2
            print(recall_votes_adv)
            print(recall_votes_normal)
            print(mean_recall_votes)
            self.set_recall_votes(mean_recall_votes)
            mean_f1_score_votes = (f1_score_normal + f1_score_adv) / 2
            self.set_f1score_votes(mean_f1_score_votes)
            mean_acc_votes = (acc_normal + acc_adv) / 2
            print(acc_adv)
            print(acc_normal)
            print(mean_acc_votes)
            raise Exception("MIRAR")
            self.set_acc_votes(mean_acc_votes)
            mean_precision_votes = (precision_normal + precision_adv) / 2
            self.set_precision_votes(mean_precision_votes)
    

    def model_predict_one_votes(self, x_sample, y_labels, adv_x_sample = None, which_model: list = None):
        print("-------->Prediction with coordinator Prob.")
        probs = []
        for idx, model in enumerate(self.models):
            adv_obj = False
            if adv_x_sample is not None:
                if idx in which_model:
                    adv_obj = True
                    prob = model.evaluate_one_proba_votes(adv_x_sample, y_labels, self.threshold)  
                    probs.append(prob)
                    print(f"Adversarial attack performed in model {idx} from models {which_model}."+
                           f"Model name: {str(model).replace('votingSystem.ADSystems.', '').split('.')[0]}")    
            if not adv_obj:
                prob = model.evaluate_one_proba_votes(x_sample, y_labels, self.threshold)
                probs.append(prob)                
            #self.free_gpu_memory(prob)
            probs_np = np.array(probs)
            means_prob = np.mean(probs_np, axis=0)
            y_preds = [1 if mean >= self.threshold else 0 for mean in means_prob]
        
        self.set_metrics(y_labels, y_preds)
        

    