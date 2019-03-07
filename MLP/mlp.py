import pandas as pd
from pandas import read_csv
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler  
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix  

def load_file(filepath):
	data_frame = read_csv(filepath, header=None, delim_whitespace =True)
	return data_frame.values

def load_features():
	features = load_file('./HARDataset/features.txt')
	return features

def load_dataset_group(group,prefix = ''):
	X = load_file(prefix+group+'/X_'+group+'.txt')
	y = load_file(prefix+group+'/y_'+group+'.txt')
	return X,y

def load_dataset(prefix=''):
	X,Y = load_dataset_group('train',prefix+'./HARDataset/')

	trainX, testX, trainy, testy = train_test_split(X, Y, test_size=0.2, random_state = 2)
	#print(testX.shape, testy.shape)

	trainy, testy = trainy[:,0], testy[:,0]
	return trainX, trainy, testX, testy

def evaluate_model (trainX,trainy, testX,testy,model):
	model.fit(trainX,trainy);
	pred = model.predict(testX);
	accuracy = accuracy_score(pred,testy);
	print (" Confusion matrix \n", confusion_matrix(testy, pred))

	return accuracy*100.0

trainX,trainy, testX, testy = load_dataset()
features = load_features()

scaler = StandardScaler()
scaler.fit(trainX)

trainX = scaler.transform(trainX)
testX = scaler.transform(testX)
mlp = MLPClassifier(hidden_layer_sizes=(30,30,30))
results = evaluate_model(trainX,trainy,testX,testy,mlp)

print('results = %.3f' %results);


