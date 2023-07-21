# Author: Nguyễn Thanh Luân

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from yellowbrick.classifier import ROCAUC

import numpy as np 
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE


#-------------------------------------------------------------------------------------------------#

class DecisionTreeClassiferModel:
    def __init__(self, data, handle_imbalance=False): 
        self.data = data
        self.handle_imbalance = handle_imbalance
        return 
    
    def split_data(self): 
        self.X = self.data.drop('decision',axis=1).values
        self.y = self.data['decision'].values
        return 
    
    def encode_data(self): 
        self.ord_en = OrdinalEncoder()
        self.X = self.ord_en.fit_transform(self.X)

        self.lb_en = LabelEncoder()
        self.y = self.lb_en.fit_transform(self.y)
        return 

    def train_test_split(self): 
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=5)
        return
    
    def Handle_Imbalance(self): 
        if self.handle_imbalance == True: 
            print('Data Oversampling is in process...')
            oversampling = SMOTE()
            self.X, self.y = oversampling.fit_resample(self.X, self.y)
        else: 
            pass
        return
        
    def get_label(self, x): 
        return self.lb_en.inverse_transform(self.y[self.y==x])[0]

    def build_model(self): 
        clf = DecisionTreeClassifier()
        clf.fit(self.X_train, self.y_train)
        label = [self.get_label(i) for i in range(4)]
        print('Model evaluation...')
        print(metrics.accuracy_score(clf.predict(self.X_test), self.y_test))
        print(metrics.classification_report(clf.predict(self.X_test), self.y_test, target_names = label))
        print(metrics.confusion_matrix(clf.predict(self.X_test), self.y_test))
        
        plt.figure(figsize = (20, 8))
        visualizer = ROCAUC(clf, classes=["acc", "good", "unacc", 'vgood'])
        visualizer.fit(self.X_train, self.y_train)        
        visualizer.score(self.X_test, self.y_test)      
        visualizer.show() 
        return 
    
    def deploy(self): 
        print('Split data to feature and target...')
        self.split_data()
        print('Encoding data...')
        self.encode_data()
        print('Split data for training-testing set....')
        self.train_test_split()
        self.Handle_Imbalance()
        print('Training Data....')
        self.build_model()

#-------------------------------------------------------------------------------------------------#

class RandomForestClassifierModel:
    def __init__(self, data, handle_imbalance=False): 
        self.data = data
        self.handle_imbalance = handle_imbalance
        return 
    
    def split_data(self): 
        self.X = self.data.drop('decision',axis=1).values
        self.y = self.data['decision'].values
        return 
    
    def encode_data(self): 
        self.ord_en = OrdinalEncoder()
        self.X = self.ord_en.fit_transform(self.X)

        self.lb_en = LabelEncoder()
        self.y = self.lb_en.fit_transform(self.y)
        return 
    
    def train_test_split(self): 
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=5)
        return
    
    def Handle_Imbalance(self): 
        if self.handle_imbalance == True: 
            print('Data Oversampling is in process...')
            oversampling = SMOTE()
            self.X, self.y = oversampling.fit_resample(self.X, self.y)
        else: 
            pass
        return
    
    def get_label(self, x): 
        return self.lb_en.inverse_transform(self.y[self.y==x])[0]
    
    def build_model(self): 
        clf = RandomForestClassifier()
        clf.fit(self.X_train, self.y_train)
        label = [self.get_label(i) for i in range(4)]
        print('Model evaluation...')
        print(metrics.accuracy_score(clf.predict(self.X_test), self.y_test))
        print(metrics.classification_report(clf.predict(self.X_test), self.y_test, target_names = label))
        print(metrics.confusion_matrix(clf.predict(self.X_test), self.y_test))
        
        plt.figure(figsize = (20, 8))
        visualizer = ROCAUC(clf, classes=["acc", "good", "unacc", 'vgood'])
        visualizer.fit(self.X_train, self.y_train)        
        visualizer.score(self.X_test, self.y_test)      
        visualizer.show() 
        return 
    
    def deploy(self): 
        print('Split data to feature and target...')
        self.split_data()
        print('Encoding data...')
        self.encode_data()
        print('Split data for training-testing set....')
        self.train_test_split()
        self.Handle_Imbalance()
        print('Training Data....')
        self.build_model()
        
#-------------------------------------------------------------------------------------------------#

class CategoricalNBModel:
    def __init__(self, data, handle_imbalance=False): 
        self.data = data
        self.handle_imbalance = handle_imbalance
        return 
    
    def split_data(self): 
        self.X = self.data.drop('decision',axis=1).values
        self.y = self.data['decision'].values
        return 
    
    def encode_data(self): 
        self.ord_en = OrdinalEncoder()
        self.X = self.ord_en.fit_transform(self.X)

        self.lb_en = LabelEncoder()
        self.y = self.lb_en.fit_transform(self.y)
        return 
    
    def train_test_split(self): 
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=5)
        return
    
    def Handle_Imbalance(self): 
        if self.handle_imbalance == True: 
            print('Data Oversampling is in process...')
            oversampling = SMOTE()
            self.X, self.y = oversampling.fit_resample(self.X, self.y)
        else: 
            pass
        return
    
    def get_label(self, x): 
        return self.lb_en.inverse_transform(self.y[self.y==x])[0]
    
    def build_model(self): 
        clf = CategoricalNB()
        clf.fit(self.X_train, self.y_train)
        label = [self.get_label(i) for i in range(4)]
        print('Model evaluation...')
        print(metrics.accuracy_score(clf.predict(self.X_test), self.y_test))
        print(metrics.classification_report(clf.predict(self.X_test), self.y_test, target_names = label))
        print(metrics.confusion_matrix(clf.predict(self.X_test), self.y_test))
        
        plt.figure(figsize = (20, 8))
        visualizer = ROCAUC(clf, classes=["acc", "good", "unacc", 'vgood'])
        visualizer.fit(self.X_train, self.y_train)        
        visualizer.score(self.X_test, self.y_test)      
        visualizer.show() 
        return 
    
    def deploy(self): 
        print('Split data to feature and target...')
        self.split_data()
        print('Encoding data...')
        self.encode_data()
        print('Split data for training-testing set....')
        self.train_test_split()
        self.Handle_Imbalance()
        print('Training Data....')
        self.build_model()
#-------------------------------------------------------------------------------------------------#

class KNNclfModel: 
    
    def __init__(self, data, handle_imbalance=False, distance = 'euclid'): 
        self.data = data
        self.handle_imbalance = handle_imbalance
        self.distance = distance
        return 
    
    def split_data(self): 
        self.X = self.data.drop('decision',axis=1).values
        self.y = self.data['decision'].values
        return 
    
    def encode_data(self): 
        self.ord_en = OrdinalEncoder()
        self.X = self.ord_en.fit_transform(self.X)

        self.lb_en = LabelEncoder()
        self.y = self.lb_en.fit_transform(self.y)
        return 
    
    def train_test_split(self): 
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=5)
        return
    
    def Handle_Imbalance(self): 
        if self.handle_imbalance == True: 
            print('Data Oversampling is in process...')
            oversampling = SMOTE()
            self.X, self.y = oversampling.fit_resample(self.X, self.y)
        else: 
            pass
        return
    
    def get_label(self, x): 
        return self.lb_en.inverse_transform(self.y[self.y==x])[0]
    
    def choose_param(self): 
        result_uniform_ma = []
        result_uniform_eu = []
        result_distance_ma = []
        result_distance_eu = []
        
        for k in range(1, 18): 
            model1 = KNeighborsClassifier(n_neighbors = k, weights='uniform', p=1)
            model1.fit(self.X_train, self.y_train)
            result_uniform_ma.append(metrics.accuracy_score(model1.predict(self.X_test), self.y_test))
            
            model2 = KNeighborsClassifier(n_neighbors = k, weights='uniform', p=2)
            model2.fit(self.X_train, self.y_train)
            result_uniform_eu.append(metrics.accuracy_score(model2.predict(self.X_test), self.y_test))
            
            model3 = KNeighborsClassifier(n_neighbors = k, weights='distance', p=1)
            model3.fit(self.X_train, self.y_train)
            result_distance_ma.append(metrics.accuracy_score(model3.predict(self.X_test), self.y_test))
            
            model4 = KNeighborsClassifier(n_neighbors = k, weights='distance', p=2)
            model4.fit(self.X_train, self.y_train)
            result_distance_eu.append(metrics.accuracy_score(model4.predict(self.X_test), self.y_test))
            
        plt.figure(figsize = (20, 8))
        plt.plot(np.arange(1, 18), result_uniform_ma)
        plt.plot(np.arange(1, 18), result_uniform_eu)
        plt.plot(np.arange(1, 18), result_distance_ma)
        plt.plot(np.arange(1, 18), result_distance_eu)
        plt.legend(['uniform_manhattan', 'uniform_euclid', 'distance_manhattan', 'distance_euclid'])
        plt.show()
        return
    
    def build_model(self): 
        k = int(input('Number of neighbors: '))
        weight = input('Choose a method for weight: ')
        p = int(input('Choose a method for calculate distance: '))
        
        clf = KNeighborsClassifier(n_neighbors = k, weights = weight, p = p)
        clf.fit(self.X_train, self.y_train)
        label = [self.get_label(i) for i in range(4)]
        print('Model evaluation...')
        print(metrics.accuracy_score(clf.predict(self.X_test), self.y_test))
        print(metrics.classification_report(clf.predict(self.X_test), self.y_test, target_names = label))
        print(metrics.confusion_matrix(clf.predict(self.X_test), self.y_test))
        
        plt.figure(figsize = (20, 8))
        visualizer = ROCAUC(clf, classes=["acc", "good", "unacc", 'vgood'])
        visualizer.fit(self.X_train, self.y_train)        
        visualizer.score(self.X_test, self.y_test)      
        visualizer.show() 
        return 
    
    def deploy(self): 
        print('Split data to feature and target...')
        self.split_data()
        print('Encoding data...')
        self.encode_data()
        print('Split data for training-testing set....')
        self.train_test_split()
        self.Handle_Imbalance()
        print('Selecting the appropriate parameter is in progress....')
        self.choose_param()
        print('Training Data....')
        self.build_model()