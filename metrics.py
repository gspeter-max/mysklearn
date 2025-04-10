import  numpy as np 
from typing import Union

class metrics : 
    
    def roc_auc_score(self,y_true : Union[list,np.ndarray],y_pred_proba : Union[list,np.ndarray],return_fpr_tpr : bool):
        
        thresholds = np.sort(np.unique(y_pred_proba))[::-1]
        tpr = [] 
        fpr = [] 
        for threshold in thresholds: 
            y_pred = (y_pred_proba > threshold).astype(int)
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            tn = np.sum((y_true == 0) & (y_true == 0))
            fp = np.sum((y_true == 0) & (y_pred == 1))


            tpr_ = tp / ((tp + fn) + 1e-6)
            tpr.append(tpr_)    
            fpr_ = fp / ((fp + tn) + 1e-6)
            fpr.append(fpr_)

        tpr = np.array(tpr)
        fpr = np.array(fpr)

        sort_index = np.argsort(tpr)
        sorted_fpr = fpr[sort_index]
        sorted_tpr = tpr[sort_index]

        roc =0.0 
        for i in range(1,len(sort_index)): 
            tini_fpr_change = fpr[i] - fpr[i-1]
            tini_tpr_change = (tpr[i] + tpr[i-1]) / 2 
            roc +=  tini_fpr_change * tini_tpr_change 

        if return_fpr_tpr is True : 
            return  sorted_tpr, sorted_fpr,roc 
        return roc 
    

        