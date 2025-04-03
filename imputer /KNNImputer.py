import numpy as np
import pandas as pd
from typing import Union
from scipy.stats import mode
class Knnimputer:

    def __init__(self,col : Union[np.ndarray,list] , k : Union[int,float] , strategy : str):
            self.k = k
            self.strategy  = strategy
            self.col = col
            self.closest_values = []

    def compute_values(self,col,imput_values):
        i = 0
        temp_list = []
        for values in col:
            if pd.isna(values):
                i += 1
                temp_list.append(imput_values[i-1])
            else:
                temp_list.append(values)

        return  np.array(temp_list)

    def call(self):
        assert len(self.col) >= self.k , ValueError(' k must be equal or greater than len(col)')
        if len(self.col) == self.k:
            global_mean = np.mean(self.col)
            return np.where(np.isnan(self.col) , global_mean,self.col)

        first_index = int(np.floor(self.k/ 2))
        last_index = int(np.ceil(self.k / 2))

        for index , values in enumerate(self.col):

            if pd.isna(values):
                if index >= first_index:
                    x__ = (len(self.col)-1) - ( index + last_index )
                    if x__ >= 0:
                        self.closest_values.append(np.concatenate([self.col[index - first_index :index],
                            self.col[index + 1 : index + 1 + last_index]
                            ]))
                    else:
                        x___ = len(self.col)- ( index + 1 )
                        if x__ == 0 :
                            self.closest_values.append(self.col[index - ( self.k - x__ ): index])
                        else:
                            self.closest_values.append(np.concatenate([self.col[ index - ( self.k - x___)  : index ],self.col[index + 1:]]))
                else:
                    x_ = index - first_index
                    self.closest_values.append(np.concatenate([self.col[:first_index + x_],self.col[index + 1 : index + 1 +(last_index - x_) ]]))

             # write that like list have that list of value  of k neighbors
            # diff null values have diff - diff values
        if self.strategy == 'mean':
            means = np.mean(self.closest_values, axis = 1)  
            return self.compute_values(self.col, means)

        elif self.strategy == 'mode':
            print(self.closest_values)

            ''' 
            pd.isna ---> changed for categoricaly fix that 
            create own mode functon and 
            create another methods 

            ''' 
            
            modes = mode(self.closest_values,axis = 1)
            return self.compute_values(self.col, modes)
        else:
            raise ValueError(f"{self.strategy} is not added over here")


structured_arr = np.arange(100, dtype=float)
structured_arr[::5] = np.nan

# structured_list = [float('nan') if i % 5 == 0 else i for i in range(100)]
mixed_list = ["apple", None, "banana", "orange", None, "grape"]
imputer = Knnimputer(mixed_list,k = 3,strategy = 'mode')
result = imputer.call()
print(result)





'''
---------------------------------------------------------
index ==== 3 

1. 3 > 2 --> 
x__ = 6 - ( 3 + 3 ) = 0 

first ===== [ 1 , 2 ] 
second ==== [ 4, 5,6] 

---------------------------------------------------------------------

index ==== 4 

4 > 2 
x__ = 6 - ( 4 + 3 ) = -1 --> move to else 
x ___ = 7 - ( 4 + 1) = 2 
first === 4 - ( 5 - 2 ) =========== [1,2,3] 
second == 4 + 1 = 5 == [ 5,6] 

pass 
--------------------------------------------------------
index == 5 

x ___ = 7 - 6 = 1 
first == 5 - ( 5 - 1 ) = 1 == [1,2,3,4] 
second == 6 

---------------------------------------------------------

index === 6 

6 > 2 
x__ = -1 ----> move to else 

x___ = 7 - ( 6 + 1 ) == 0 
first === 6 - ( 5 + 0) = 1 == [ 1,2,3,4,5] 
second == ( 6 + 1 




k == 6 with len == 7 
-----------------------------------------
index == 0 
else 
x_ = 0 - 3 = -3 
first == 3 - 3 =0 --> [] 
second == 1 : 1 + 6 
pass 

------------------------------------------------------
index == 2 

else 
x_ = 2 - 3 == -1 
first == 3 -1 == 2 --> [ 0.1] 
second == 3 : 3 + ( 3 + 1 ) = 3 : 7 = [3,4,5,6] 

-------------------------------------------------------
index == 5 

greater 
x__ = 6 - ( 5 + 3 ) = -2 --> move else 
x___ = 7 - ( 5 + 1) == 1 not == 0 ---> move else
first == 5 - ( 6 - 1 ) ==  0 : 5 = [0,1,2,3,4] 
second == [6] 
pass 
''' 
