
from pandas import read_csv
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import numpy as np
from matplotlib import pyplot as plt

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
	trainX, trainy = load_dataset_group('train',prefix+'./HARDataset/')
	#print(trainX.shape, trainy.shape)

	testX,testy = load_dataset_group('test','./HARDataset/')
	#print(testX.shape, testy.shape)

	trainy, testy = trainy[:,0], testy[:,0]
	return trainX, trainy, testX, testy

def evaluate_model (trainX,trainy, testX,testy,model):
	model.fit(trainX,trainy);
	pred = model.predict(testX);
	accuracy = accuracy_score(pred,testy);
	return accuracy*100.0

trainX,trainy, testX, testy = load_dataset()
features = load_features()
model = RandomForestClassifier(n_estimators =95,criterion = "entropy",max_features = "log2")
sfm = SelectFromModel(model,threshold = 0.000011)
sfm.fit(trainX,trainy)
trainX_imp = sfm.transform(trainX);
testX_imp = sfm.transform(testX);
results = evaluate_model(trainX_imp,trainy,testX_imp,testy,model)

# for name, importance in zip(features, model.feature_importances_):
# 	print(name, "=", importance)

# plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
# plt.show()

print("prarameters in use: ",model.get_params());
print('results = %.3f' %results);




