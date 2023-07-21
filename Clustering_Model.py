# Author: Nguyễn Thanh Luân

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn import preprocessing
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn import metrics

from OutlierDetection import ISOLATION_FOREST_REMOVE, LOCAL_OUTLIER_REMOVE

import warnings
warnings.filterwarnings('ignore')

#-------------------------------------------------------------------------------------------------#
def display_elbow(elbow): 
    plt.plot(range(1,10), elbow)
    plt.title('The Elbow Method')
    plt.xlabel('no of clusters')
    plt.ylabel('elbow')
    plt.show()

def visualize_result(array, cluster):
    fig = plt.figure(figsize = (20, 10))
    ax = plt.axes(projection='3d')
    ax.scatter(array[:, 0], array[:, 1], array[:, 2], c=cluster, s=50, cmap='viridis')
    plt.show()
#-------------------------------------------------------------------------------------------------#

class KMEAN:
    def __init__(self, data, features = ['price', 'mileage', 'tax'], method_scaler = 'log10', model_outlier = 'iso', method_elbow = True, method_silhouete = True): 
        self.data = data 
        self.features = features
        self.method_scaler = method_scaler
        self.model_outlier = model_outlier
        self.method_elbow = method_elbow
        self.method_silhouete = method_silhouete
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
        return
    
    def remove_outlier_func(self):
        self.X = self.data.drop('price', axis=1).values
        self.y = self.data['price'].values
        if self.model_outlier == 'iso': 
            model = ISOLATION_FOREST_REMOVE(self.X, self.y, contamination=0.1)
            idx, X_new, y_new = model.remove_outlier()
        elif self.model_outlier == 'local': 
            model = LOCAL_OUTLIER_REMOVE(self.X, self.y)
            idx, X_new, y_new = model.remove_outlier()
            
        self.X = X_new
        self.y = y_new
        self.data = pd.concat([pd.DataFrame(self.X), pd.DataFrame(self.y)], axis=1).reset_index(drop=True)
        self.data.columns = ['Brand', 'model', 'year', 'transmission', 'mileage',
                           'fuelType', 'tax', 'mpg', 'engineSize', 'price']
        return 
    
    def select_feature(self):
        self.array = self.data[self.features].values
        return 
    
    def elbow_method(self): 
        if self.method_elbow == True:
            print('Calculate number of cluster with elbow method')
            elbow = []
            for i in range(1, 10):
                kmeans = KMeans(n_clusters=i, init='k-means++', n_init=100)
                kmeans.fit(self.array)
                elbow.append(kmeans.inertia_)
            display_elbow(elbow)
        else: 
            pass
        return 
    
    def silhouette_method(self):
        if self.method_silhouete == True:
            print('Calculate number of cluster with sihouette scoring')
            fig, ax = plt.subplots(3, 2, figsize = (15,8)) 
            for n_clusters in range(2,8):
                kmeans = KMeans(init='k-means++', n_clusters = n_clusters, n_init=100)
                q, mod = divmod(n_clusters, 2)
                kmeans.fit(self.array)
                clusters = kmeans.predict(self.array)
                silhouette_avg = metrics.silhouette_score(self.array, clusters)
                print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)
                # ----- # 
                kmeans = KMeans(init='k-means++', n_clusters = n_clusters, n_init=100)
                visualizer = SilhouetteVisualizer(kmeans, Colors = 'yellowbrick', ax = ax[q-1][mod]) 
                visualizer.fit(self.array)
            visualizer.show()
        else:
            pass
        return
    
    def build_model(self):
        n_cluster = int(input('Number of clusters: '))
        kmeans = KMeans(init='k-means++', n_clusters = n_cluster, n_init=100)
        kmeans.fit(self.array)
        clusters = kmeans.predict(self.array)
        visualize_result(self.array, clusters)
        return 
    
    def deploy(self): 
        print('Scaling Data....')
        self.scaler_data()
        print('Encoding Data....')
        self.encode_data()
        print('Remove Outlier....')
        self.remove_outlier_func()
        print('Select feature to training....') 
        self.select_feature()
        self.elbow_method()
        self.silhouette_method()
        print('Training Data....')
        self.build_model()