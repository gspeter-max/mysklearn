 
import numpy as np
from scipy.spatial.distance import cdist 


class Nearest_Neighbors:

    def __init__(self, k):
        self.k = k 
    
    def predict(self,x1,x2 = None,return_index : bool = True):
        if x2 is None: 
            x2 = x1  
        distance = cdist(x1,x2,metric = 'euclidean')
        if x2 is None: 
            np.fill_diagonal(distance , np.inf)
        index = np.argpartition(distance, self.k -1, axis = 1 )[:,:self.k]
        # FIRST OPTION  
        # diff = np.take_along_axis(distance, index, axis = 1)
        index_ = np.arange(distance.shape[0])[:,None]
        diff= distance[index_, index]
        if return_index is True: 
            return distance, index          
        return distance



