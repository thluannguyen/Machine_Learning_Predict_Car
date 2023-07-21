# Author: Nguyễn Thanh Luân

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
#-------------------------------------------------------------------------------------------------#
"""
This module will be used to remove some outlier data point in training dataset
Input data set is x_train 2D array and y_train 2D array
"""
#-------------------------------------------------------------------------------------------------#
class ISOLATION_FOREST_REMOVE():
    
    def __init__(self, x_train, y_train, contamination=0.1):
        self.name = "isolation_forest"
        self.x_train = x_train
        self.y_train = y_train
        self.contamination = contamination
        self.model = IsolationForest(contamination=self.contamination)

    def remove_outlier(self):
        y_iso = self.model.fit_predict(self.x_train)
        in_forest_idx = (y_iso != -1)
        x_in_forest = self.x_train[in_forest_idx, :]
        y_in_forest = self.y_train[in_forest_idx]
        print(f'Data source have contain {100-round(len(x_in_forest)/len(self.x_train)*100)}% record is outlier')
        return in_forest_idx, x_in_forest, y_in_forest

#-------------------------------------------------------------------------------------------------#
class LOCAL_OUTLIER_REMOVE():

    def __init__(self, x_train, y_train):
        self.name = "local_outlier_factor"
        self.x_train = x_train
        self.y_train = y_train
        self.model = LocalOutlierFactor()

    def remove_outlier(self):
        y_lof = self.model.fit_predict(self.x_train)
        in_local_idx = (y_lof != -1)
        x_in_local = self.x_train[in_local_idx, :]
        y_in_local = self.y_train[in_local_idx]
        print(f'Data source have contain {100-round(len(x_in_local)/len(self.x_train)*100)}% record is outlier')
        return in_local_idx, x_in_local, y_in_local