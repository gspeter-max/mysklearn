from scipy.spatial.distance import cdist
import scipy 
import numpy as np

class KNNClassification:

    def __init__(self, k=3):
        self.k = k
        self.x_train = None
        self.y_train = None

    def fit(self, x_train, y_train):
        self.x_train = np.array(x_train) 
        self.y_train = np.array(y_train).reshape(-1,1)


    def predict(self, x_test):

        distance = cdist( x_test,self.x_train , metric = 'cosine')
        index = np.argpartition(distance,self.k -1, axis = 1)[:,:self.k]
        predict = np.take_along_axis(self.y_train.T,index, axis = 1)
        pred = scipy.stats.mode(predict, axis = 1)
        return pred
