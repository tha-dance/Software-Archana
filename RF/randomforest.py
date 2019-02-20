from pandas import read_csv
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

def load_file(filepath):
	data_frame = read_csv(filepath, header=None, delim_whitespace =True)
	return data_frame.values

def load_dataset_group(group,prefix = ''):
	X = load_file(prefix+group+'/X_'+group+'.txt')
	y = load_file(prefix+group+'/y_'+group+'.txt')
	return X,y

def load_dataset(prefix=''):
	trainX, trainy = load_dataset_group('train',prefix+'./HARDataset/')
	print(trainX.shape, trainy.shape)

	testX,testy = load_dataset_group('test','./HARDataset/')
	print(testX.shape, testy.shape)

	trainy, testy = trainy[:,0], testy[:,0]
	return trainX, trainy, testX, testy

def evaluate_model (trainX,trainy, testX,testy,model):
	model.fit(trainX,trainy);
	pred = model.predict(testX);
	accuracy = accuracy_score(pred,testy);
	return accuracy*100.0

trainX,trainy, testX, testy = load_dataset()
model = RandomForestClassifier(n_estimators =97,criterion = "entropy",max_features = "log2")
results = evaluate_model(trainX,trainy,testX,testy,model)
print("prarameters in use: ",model.get_params());
print('results = %.3f' %results);