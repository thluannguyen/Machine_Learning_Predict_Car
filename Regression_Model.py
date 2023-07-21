# Author: Nguyễn Thanh Luân

import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt

from sklearn import preprocessing 
from sklearn import metrics
from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SequentialFeatureSelector as SFS

from OutlierDetection import ISOLATION_FOREST_REMOVE, LOCAL_OUTLIER_REMOVE

#-------------------------------------------------------------------------------------------------#

## statistical function to calculate performance model 
def evaluate_model(true, predicted):  
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    mape = metrics.mean_absolute_percentage_error(true, predicted)
    print('========================================================')
    print('MAE:', mae)
    print('MSE:', mse)
    print('RMSE:', rmse)
    print('R2-Square:', r2_square)
    print('MAPE:', mape)
    print('========================================================')
 #-------------------------------------------------------------------------------------------------#   

class LinearRegressionModel: 
    def __init__(self, data, method_scaler = 'log10', model_outlier='iso', imple_SF = False,test_size=0.2, random_state=32): 
        self.data = data
        self.method_scaler = method_scaler
        self.model_outlier = model_outlier
        self.imple_SF = imple_SF
        self.test_size = test_size
        self.random_state = random_state
        self.columns_scaler = ['price', 'mileage', 'tax']
        return 
    
    def scaler_data(self): 
        if self.method_scaler == 'log10': 
            for col in self.columns_scaler: 
                if col != 'tax': 
                    self.data[col] = np.log(self.data[col])+1
                else: 
                    self.data[col] = np.log(self.data[col]+2)
            return 
        elif self.method_scaler == 'minmax': 
            scaler = preprocessing.MinMaxScaler()
        elif self.method_scaler == 'standard': 
            scaler = preprocessing.StandardScaler()
        for col in self.columns_scaler:
            self.data[col] = scaler.fit_transform(np.array(self.data[col]).reshape(-1, 1))
        else: 
            print('Please choose appropiate method!!')
        return
    
    def encode_data(self): 
        le = preprocessing.LabelEncoder()
        for col in self.data.select_dtypes('O'): 
            self.data[col] = le.fit_transform(self.data[col])
                
        print('Spliting data after encoding data')
        self.X = self.data.drop(['price', 'tax', 'mpg'],axis=1).values
        self.y = self.data['price'].values
        return
    
    def remove_outlier_func(self):
        if self.model_outlier == 'iso': 
            model = ISOLATION_FOREST_REMOVE(self.X, self.y, contamination=0.1)
            idx, X_new, y_new = model.remove_outlier()
        elif self.model_outlier == 'local': 
            model = LOCAL_OUTLIER_REMOVE(self.X, self.y)
            idx, X_new, y_new = model.remove_outlier()
            
        self.X = X_new
        self.y = y_new
        return 
    
    def train_test_split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, 
                                            test_size=self.test_size, random_state=self.random_state)
        return 
    
    def select_feature(self): 
        plt.figure(figsize = (20, 8))
        data_corr = pd.concat([pd.DataFrame(self.X), pd.DataFrame(self.y)],axis=1)
        data_corr.columns = ['Brand', 'model', 'year', 'transmission', 'mileage', 'fuelType', 'engineSize', 'price']
        sns.heatmap(data_corr.corr(), annot = True)
        plt.show()

        if self.imple_SF == False: 
            return 
        else: 
            print('Deploy the feature search model, but need human observation to select the feature according to the need for resource saving or accurate prediction.')
            for nf in range(1, 9): 
                print('Number of features:', nf)
                l_reg = LinearRegression()
                sfs = SFS(l_reg,
                          n_features_to_select=nf,
                          direction='backward',
                          scoring = 'neg_mean_absolute_percentage_error')
                sfs.fit(self.X, self.y)
                print(sfs.get_support())
                feature_new = sfs.transform(self.X)
                l_reg.fit(feature_new, self.y)
                print(evaluate_model(self.y, l_reg.predict(feature_new)))

            ## with 9 features
            print('Number of features: 9')
            l_reg = LinearRegression()
            l_reg.fit(self.X, self.y)
            print(evaluate_model(self.y, l_reg.predict(self.X)))
        
        return 
        
    def build_model(self):
        print('Training model linear regression with method polynomial features.....')
        poly_reg = preprocessing.PolynomialFeatures(degree = 4)
        X_poly = poly_reg.fit_transform(self.X_train)
        model = LinearRegression()
        model.fit(X_poly, self.y_train)
        X_poly_test = poly_reg.fit_transform(self.X_test)
        y_pred = model.predict(X_poly_test)
        print(evaluate_model(self.y_test, y_pred))
    
    def deploy(self): 
        print('Scaling Data....')
        self.scaler_data()
        print('Encoding Data....')
        self.encode_data()
        print('Remove Outlier....')
        self.remove_outlier_func()
        print('Split data for training-testing set....') 
        self.train_test_split()
        print('Select Feature....')
        self.select_feature()
        print('Training Data....')
        self.build_model()
    
#-------------------------------------------------------------------------------------------------#    
class RandomForestRegressorModel: 
    def __init__(self, data, method_scaler = 'log10', model_outlier='iso',test_size=0.2, random_state=32): 
        self.data = data
        self.method_scaler = method_scaler
        self.model_outlier = model_outlier
        self.test_size = test_size
        self.random_state = random_state
        self.columns_scaler = ['price', 'mileage', 'tax']
    
    def scaler_data(self): 
        if self.method_scaler == 'log10': 
            for col in self.columns_scaler: 
                if col != 'tax': 
                    self.data[col] = np.log(self.data[col])+1
                else: 
                    self.data[col] = np.log(self.data[col]+2)
            return 
        elif self.method_scaler == 'minmax': 
            scaler = preprocessing.MinMaxScaler()
        elif self.method_scaler == 'standard': 
            scaler = preprocessing.StandardScaler()
        for col in self.columns_scaler:
            self.data[col] = scaler.fit_transform(np.array(self.data[col]).reshape(-1, 1))
        else: 
            print('Please choose appropiate method!!')
        return
    
    def encode_data(self): 
        le = preprocessing.LabelEncoder()
        for col in self.data.select_dtypes('O'): 
            self.data[col] = le.fit_transform(self.data[col])
                
        print('Spliting data after encoding data')
        self.X = self.data.drop('price',axis=1).values
        self.y = self.data['price'].values
        return
    
    def remove_outlier_func(self):
        if self.model_outlier == 'iso': 
            model = ISOLATION_FOREST_REMOVE(self.X, self.y, contamination=0.1)
            idx, X_new, y_new = model.remove_outlier()
        elif self.model_outlier == 'local': 
            model = LOCAL_OUTLIER_REMOVE(self.X, self.y)
            idx, X_new, y_new = model.remove_outlier()
            
        self.X = X_new
        self.y = y_new
        return 
    
    def train_test_split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, 
                                            test_size=self.test_size, random_state=self.random_state)
        return 
    
    def build_model(self):
        model = RandomForestRegressor()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        print(evaluate_model(self.y_test, y_pred))        
    
    def deploy(self): 
        print('Scaling Data....')
        self.scaler_data()
        print('Encoding Data....')
        self.encode_data()
        print('Remove Outlier....')
        self.remove_outlier_func()
        print('Split data for training-testing set....') 
        self.train_test_split()
        print('Training Data....')
        self.build_model()        
    

#-------------------------------------------------------------------------------------------------#    
class DecisionTreeRegressorModel: 
    def __init__(self, data, method_scaler = 'log10', model_outlier='iso',test_size=0.2, random_state=32): 
        self.data = data
        self.method_scaler = method_scaler
        self.model_outlier = model_outlier
        self.test_size = test_size
        self.random_state = random_state
        self.columns_scaler = ['price', 'mileage', 'tax']
    
    def scaler_data(self): 
        if self.method_scaler == 'log10': 
            for col in self.columns_scaler: 
                if col != 'tax': 
                    self.data[col] = np.log(self.data[col])+1
                else: 
                    self.data[col] = np.log(self.data[col]+2)
            return 
        elif self.method_scaler == 'minmax': 
            scaler = preprocessing.MinMaxScaler()
        elif self.method_scaler == 'standard': 
            scaler = preprocessing.StandardScaler()
        for col in self.columns_scaler:
            self.data[col] = scaler.fit_transform(np.array(self.data[col]).reshape(-1, 1))
        else: 
            print('Please choose appropiate method!!')
        return
    
    def encode_data(self): 
        le = preprocessing.LabelEncoder()
        for col in self.data.select_dtypes('O'): 
            self.data[col] = le.fit_transform(self.data[col])
                
        print('Spliting data after encoding data')
        self.X = self.data.drop('price',axis=1).values
        self.y = self.data['price'].values
        return
    
    def remove_outlier_func(self):
        if self.model_outlier == 'iso': 
            model = ISOLATION_FOREST_REMOVE(self.X, self.y, contamination=0.1)
            idx, X_new, y_new = model.remove_outlier()
        elif self.model_outlier == 'local': 
            model = LOCAL_OUTLIER_REMOVE(self.X, self.y)
            idx, X_new, y_new = model.remove_outlier()
            
        self.X = X_new
        self.y = y_new
        return 
    
    def train_test_split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, 
                                            test_size=self.test_size, random_state=self.random_state)
        return 
    
    def build_model(self):
        model = DecisionTreeRegressor()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        print(evaluate_model(self.y_test, y_pred))        
    
    def deploy(self): 
        print('Scaling Data....')
        self.scaler_data()
        print('Encoding Data....')
        self.encode_data()
        print('Remove Outlier....')
        self.remove_outlier_func()
        print('Split data for training-testing set....') 
        self.train_test_split()
        print('Training Data....')
        self.build_model()         