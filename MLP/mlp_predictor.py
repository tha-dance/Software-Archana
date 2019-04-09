import pandas as pd
from pandas import read_csv
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler  
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.model_selection import cross_val_score 
from sklearn.externals import joblib

def load_file(filepath):
	data_frame = read_csv(filepath, header=None, delim_whitespace =True)
	return data_frame.values

def load_dataset_group(filename):
	X = load_file(filename)
	return X;

testX = load_dataset_group('./test/chicken.txt')
scaler = StandardScaler()
testX = scaler.transform(testX)
model = joblib.load("mlp_final")
pred = model.predict(testX)
print(pred);