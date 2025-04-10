import numpy as np

class xgbclassifier: 
    
    def __init__(self): 
        pass 

    def sigmoid(self,x): 
        return 1 / ( 1 + np.exp(-x)) 
    
    def initial_prediciton(self,y): 
        y = np.mean(y) 
        log_logist = np,log(y / (1 - y)) 
        return  self.sigmoid(log_logist) 
    
    def 