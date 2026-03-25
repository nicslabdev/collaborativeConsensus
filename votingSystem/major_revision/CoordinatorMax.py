import statistics

import numpy as np

from CoordinatorModel import CoordinatorModel
import gc
#import cupy as cp
from scipy import stats

class CoordinatorMax(CoordinatorModel):

    def __init__(self, models, window_size):
        CoordinatorModel.__init__(self, models, window_size)

    #def free_gpu_memory(self, preds):
    #    del preds
    #    cp.get_default_memory_pool().free_all_blocks()
    #    gc.collect()

    def set_metrics_old(self, y_real_sample, normal_preds_list, adv_preds_list, adv_y_sample = None, which_model = None):
        # Recall of coordinator - adversarial
        recall_votes_adv  = 0.0
        recall_votes_normal = 0.0
        f1_score_adv = 0.0
        f1_score_normal = 0.0
        acc_adv = 0.0
        acc_normal = 0.0
        precision_adv = 0.0
        precision_normal = 0.0
        if adv_y_sample is not None:
            fixed_adv_preds_list = [arr.reshape(-1) if arr.ndim > 1 else arr for arr in adv_preds_list]
            adv_preds_array = np.array(adv_preds_list)
            y_preds_adv = stats.mode(adv_preds_array, axis=0)[0] #np.max(adv_preds_array, axis=0)    
            recall_votes_adv = self.get_recall_votes(adv_y_sample, y_preds_adv)
            f1_score_adv = self.get_f1score_votes(adv_y_sample, y_preds_adv)
            acc_adv  = self.get_acc_votes(adv_y_sample, y_preds_adv)
            precision_adv = self.get_precision_votes(adv_y_sample, y_preds_adv)
        # Recall of coordinator - normal
        if which_model is None or len(which_model) < 5:
            fixed_normal_preds_list = [arr.reshape(-1) if arr.ndim > 1 else arr for arr in normal_preds_list]
            #raise Exception("eee")
            normal_preds_array = np.array(normal_preds_list)   
            y_preds_normal = stats.mode(normal_preds_array, axis=0)[0] #np.max(normal_preds_array, axis=0) 
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
            mean_recall_votes = (recall_votes_normal + recall_votes_adv) / 2
            self.set_recall_votes(mean_recall_votes)
            mean_f1_score_votes = (f1_score_normal + f1_score_adv) / 2
            self.set_f1score_votes(mean_f1_score_votes)
            mean_acc_votes = (acc_normal + acc_adv) / 2
            self.set_acc_votes(mean_acc_votes)
            mean_precision_votes = (precision_normal + precision_adv) / 2
            self.set_precision_votes(mean_precision_votes)

    def model_predict_one_votes(self, x_sample, y_labels, adv_x_sample = None, which_model: list = None):
        print("-------->Prediction with coordinator Max.")
        preds = []
        for idx, model in enumerate(self.models):
            adv_obj = False
            if adv_x_sample is not None:
                if idx in which_model:
                    adv_obj = True
                    pred = model.evaluate_one_votes(adv_x_sample, y_labels)  
                    preds.append(pred)
                    print(f"Adversarial attack performed in model {idx} from models {which_model}."+
                           f"Model name: {str(model).replace('votingSystem.ADSystems.', '').split('.')[0]}")    
                        
            if not adv_obj:
                pred = model.evaluate_one_votes(x_sample, y_labels)
                preds.append(pred)                
            #self.free_gpu_memory(prob)
        preds_np = np.array(preds)
        y_preds= stats.mode(preds_np, axis=0)[0]
        self.set_metrics(y_labels, y_preds)