"""
Script for creating de MLP model, training and saving it.

Author:Lesly Guerrero

"""
from neural_network import NeuralNetwork
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
import pickle


def get_data():
	"""Function that reads the csv file, splits and scales the data

	Returns
	-------
	np.array
		X and Y arrays for train, test and validation
	"""
	colnames = ["position_x", "position_y", "velocity_y", "velocity_x"]
	data = pd.read_csv("ce889_dataCollection.csv", names=colnames, header=None)

	X_train, X_test, y_train, y_test = train_test_split(
		np.array(data[["position_x", "position_y"]]),
		np.array(data[["velocity_x", "velocity_y"]]),
		test_size=0.3,
		random_state=42,
	)
	X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
	x_scaler = preprocessing.MinMaxScaler()
	y_scaler = preprocessing.MinMaxScaler()
	X_trainS = x_scaler.fit_transform(X_train)
	y_trainS= y_scaler.fit_transform(y_train)
	X_testS = x_scaler.transform(X_test)
	y_testS = y_scaler.transform(y_test)
	X_valS = x_scaler.transform(X_val)
	y_valS = y_scaler.transform(y_val)

	return X_trainS, X_testS,X_valS, y_trainS, y_testS, y_valS,x_scaler,y_scaler


if __name__ == "__main__":

	X_train,X_test,X_val, y_train,y_test,y_val, Xscaler,Yscaler = get_data()
	MyNN = NeuralNetwork(learning_rate_alpha=0.9,learning_rate_etha=0.0001, momentum=0.99,early_stopping=False)
	MyNN.fit(
		x=X_train,
		y=y_train,
		x_val=X_val,
		y_val=y_val,
		hidden_layer=[10],
		n_epochs=150,
		add_bias=True
	)

	MyNN.plot()

	p,rmse=MyNN.test(X=X_test,y=y_test)


	print("Min RMSE Train: ",MyNN.min_train,"Min RMSE Validation: ",MyNN.min_val)
	print("Min RMSE Train: ",MyNN.rmse,"Min RMSE Validation: ",MyNN.rmse_val)

	print("RMSE on test: ", rmse)

	#save the model a scaler objects

	# with open("red_entrenada.pickle","wb") as file:
	# 	pickle.dump(MyNN,file)

	# with open("scalador_x.pickle","wb") as file:
	# 	pickle.dump(Xscaler,file)

	# with open("scalador_y.pickle","wb") as file:
	# 	pickle.dump(Yscaler,file)
