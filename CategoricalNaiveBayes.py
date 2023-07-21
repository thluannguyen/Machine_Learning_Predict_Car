# Author: Nguyễn Thanh Luân

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.model_selection import StratifiedKFold

import warnings
warnings.filterwarnings('ignore')


df = pd.read_excel('Car_Dataset.xlsx', sheet_name = 'Info').drop('Unnamed: 0', axis=1)



class CategoricalNaiveBayes:
    # Constructor
    def __init__(self):
        self.probs = dict()
        self.cond_probs = dict()
        self.targets = list()
        self.columns = list()
    
    # Fit method
    def fit(self, x, y, column_names):
        self.__init__()
        # Preparing DataFrame
        dataset = pd.DataFrame(data=x, index=None, columns=column_names[:-1])
        target_column_name = column_names[-1]
        dataset[target_column_name] = y
        
        # Preparing probabilities dictionary
        for column in dataset:
            self.probs[column] = dict()
            for value in dataset[column].unique():
                self.probs[column][value] = len(dataset.query('{0} == @value'.format(column))) / len(dataset)
        
        # Preparing conditional_probabilities dictionary
        for column in dataset.drop([target_column_name], axis=1):
            self.cond_probs[column] = dict()
            for value1 in dataset[column].unique():
                for value2 in dataset[target_column_name].unique():
                    self.cond_probs[column][f'{value1}-{value2}'] = len(dataset.query('{0} == @value1 & {1} == @value2'.format(column, target_column_name))) / len(dataset.query('{0} == @value2'.format(target_column_name)))
        
        self.targets = dataset[target_column_name].unique()
        self.columns = column_names
    
    # Predict method
    def predict(self, x):
        predicts = list()
        for row in x:
            target_prob_dict = dict()
            for target in self.targets:
                row_cond_probs = [self.cond_probs[column][f'{value}-{target}'] for column, value in zip(self.columns, row)]
                target_prob_dict[target] = ( np.prod(row_cond_probs) * self.probs[self.columns[-1]][target] )
            predicts.append( max(target_prob_dict, key=target_prob_dict.get) )
        return predicts

columns = df.columns.to_numpy()
target = (df["decision"]).to_numpy()
del df["decision"]
data = df.to_numpy()


cv = StratifiedKFold(n_splits=5, shuffle=True)
efficency = []
for train_index, test_index in cv.split(data, target):
    train_x, test_x = data[train_index], data[test_index]
    train_y, test_y = target[train_index], target[test_index]
    nb = CategoricalNaiveBayes()
    nb.fit(train_x, train_y, columns)
    pred_y = nb.predict(test_x)
    efficency.append(accuracy_score(test_y, pred_y))

print(f"Average classification efficency (\"categ.csv\" dataset) = {np.average(efficency) * 100}%")
print(classification_report(nb.predict(test_x),test_y, target_names = ['acc', 'good', 'unacc', 'vgood']))
print(confusion_matrix(nb.predict(test_x), test_y))