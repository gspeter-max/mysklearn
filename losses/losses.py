
import numpy as np 

class losses: 
    
    def __init__(self):
        self.loss = None
    
    def mse_loss(y_true, y_pred):
        return np.mean((y_true - y_pred)**2)

    def mse_grad(y_true, y_pred):
        return 2 * (y_pred - y_true) / len(y_true)

    def log_loss(y_true, y_pred, eps=1e-15):
        y_pred = np.clip(y_pred, eps, 1 - eps) 
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def log_loss_grad(y_true, y_pred, eps=1e-15):
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return (y_pred - y_true) / (y_pred * (1 - y_pred) * len(y_true))
    
    def inital_prediction(y): 
        
            