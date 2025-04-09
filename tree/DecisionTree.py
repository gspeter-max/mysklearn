import numpy as np 

class decision_tree_classifier: 
    
    def __init__(self):
        self.residule = None 
        
    def residule(self, y_true, y_pred): 
        self.residule = y_true - y_pred 
        
    def best_split(self, x: np.ndarray): 
        for threshold in x.unique(): 
            left_index = x <= threshold 
            right_index = x > threshold
            
            if len(self.residule[left_index]) == 0 or len(self.residule[right_index] == 0 ): 
                continue
        
            mse_left = np.mean((self.residule[left_index])**2) 
            mse_right = np.mean((self.residule[right_index])**2) 
            total_loss = (mse_left * np.sum(left_index)) +  (mse_right * np.sum(right_index))
            
            if total_loss < best_loss : 
                best_loss = total_loss
                best_threshold = threshold
                best_left = left_index
                best_right  = right_index 
                
        return best_loss, best_threshold, best_left, best_right 
    def make_tree(self): 
        # to do that peter oo genius peter 
        
                
                