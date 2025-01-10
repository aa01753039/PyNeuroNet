"""
neural_network.py

This module contains the implementation of a simple neural network using a custom Neuron class.
The NeuralNetwork class supports training, testing, and prediction tasks through feedforward
and backpropagation methods.

Classes:
- NeuralNetwork: A class representing a simple neural network with one or more hidden layers.

Dependencies:
- neuron.py: Contains the Neuron class used in the NeuralNetwork.

Example Usage:
```python
from neural_network import NeuralNetwork

# Create a neural network with default hyperparameters
nn = NeuralNetwork()

# Train the neural network on a dataset with specified input, target, and hidden layer configuration
nn.fit(x_train, y_train, hidden_layer=[10, 5], n_epochs=100)

# Test the trained neural network on a separate test dataset
predictions, test_error = nn.test(x_test, y_test)

# Make predictions on new, unseen data
new_data_predictions = nn.predict(new_data)

Author: Lesly Guerrero.
Date: October 24,2023.
"""

from neuron import Neuron
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

class NeuralNetwork:
	def __init__(self, learning_rate_alpha=0.5,learning_rate_etha=0.00001, momentum=0.1,delta =0.0001,early_stopping=True)->None:
		"""Initialize the Neural Network.

		Parameters
		----------
		learning_rate_alpha : float, optional
			The learning rate for forward pass and gradient calculation, by default 0.5
		learning_rate_etha : float, optional
			The learning rate for weight updates, by default 0.00001
		momentum : float, optional
			The momentum for weight updates, by default 0.1
		delta: float, optional
			The accepted improvment value in the training loss betwen an epoch and the one before.
		early_stopping: bool, optional
			If true, the delta early stopping is activated.
		"""
		# learning rate and momentum are hyperparameters
		self.learning_rate_alpha = learning_rate_alpha #learning rate used in the feedforward and gradient
		self.learning_rate_etha = learning_rate_etha # learning rate used in weight update
		self.momentum = momentum
		self.early_s=early_stopping
		self.delta =delta #delta value used for early stopping
		self.errors = [] #initialize the errors list, this will contain the MSE for each epoch
		self.errors_val = [] #initialize the errors list, this will contain the MSE for each epoch in the validation set
		self.NN = [] #initialize the NN list, this will contain lists of neurons representing each layer of the neural network


	def mean_squared_error(self, predictions:list, targets:list)->float:
		"""Function that calculates de Mean Squared Error

		Parameters
		----------
		predictions : list
			vector of predicted outputs
		targets : list
			vector of expected outputs

		Returns
		-------
		float
			MSE: mean of the squared diference of the expected output and the predicted output
		"""

		return np.square(np.subtract(np.array(targets),np.array(predictions))).mean()

	def feedforward(self, input:np.array, i:int,out:str)->None:
		"""Perform the forward pass through the neural network.

		Parameters
		----------
		input : np.array
			Input features.
		i : int
			Index of the current input in the dataset.
		out: str
			Type of output to access and save corresponfing output
			(could be output/output_val).
		"""
		# Forward pass
		for layer in self.NN:
			new_input = [neuron.activate(input) for neuron in layer] #activate each neuron (sigmoide activation function)
			input = new_input #the inputs for the next layer are going to be the list of inputs calculated of the actual layer
		if out == "output":
			self.output[i] = input
		if out == "output_val":
			self.output_val[i] = input

	def backpropagation(self, error:float)->None:
		"""Perform the backward pass through the neural network.

		Parameters
		----------
		error : float
			Error signal(expected output-predicted ouput).
		"""
		# Back pass
		for i in reversed(range(len(self.NN))):
			layer = self.NN[i]

			if i == len(self.NN) - 1:
				for j in range(len(layer)):
					neuron = layer[j]
					neuron.backpropagate_ouput(error[j])
			elif i < len(self.NN) - 1:
				for j in range(len(layer)):
					next_layer_weights = []
					next_layer_gradient = []
					neuron = layer[j]
					for next_layer_neuron in self.NN[i + 1]:
						next_layer_weights.append(next_layer_neuron.weights[j])
						next_layer_gradient.append(next_layer_neuron.gradient)
					neuron.backpropagate_hidden(next_layer_gradient, next_layer_weights)

		for layer in self.NN:
				for neuron in layer:
					neuron.update()
	
	def early_stopping(self,x:np.array)->bool:
		""" If having information of two or more epochs, 
		the difference betwen 2 last training loss calculations is compared with the delta value.
		
		Parameters
		----------
		x : np.array
			Array of training loss across epochs.
		
		Returns
		-------
		bool
			Wether activate or not this early srtopping criteria.

		"""
		if len(x) > 2:
			self.dif = x[-2] - x[-1]
			if  self.dif < self.delta:
				return True
		return False
		

	def fit(self, x: list, y: list,x_val:list,y_val:list, hidden_layer: list, n_epochs=100, add_bias=True)->float:
		""" Create the network architecture (list of lists containing Neuron objects),
		  	Train the neural network using online training in the backpropagation,
			Testing updated weights on the validation tests.

		Parameters
		----------
		x : list
			Input data.
		y : list
			Target outputs.
		x_val: list
			Input data used for validation.
		y_val: list
			Target outputs used for validation.
		hidden_layer : list
			List specifying the number of neurons in each hidden layer.
		n_epochs : int, optional
			Number of training epochs, by default 100
		add_bias : bool, optional
			Whether to include a bias in the neuron's computation, by default True

		"""
		self.epochs = n_epochs
		self.inputs = [np.array(xi) for xi in x]
		self.output = [0] * len(self.inputs)
		self.inputs_val = [np.array(xi) for xi in x_val]
		self.output_val = [0] * len(self.inputs_val)
		self.expected_outputs = [np.array(yi) for yi in y]
		self.expected_outputs_val = [np.array(yi) for yi in y_val]
		self.architecture = (
			[len(self.inputs[0])] + hidden_layer + [len(self.expected_outputs[0])]
		)

		for i in range(1, len(self.architecture)):
			layer = [
				Neuron(
					n_input=self.architecture[i - 1], #using the atributes of the prev layer
					learning_rate_alpha=self.learning_rate_alpha,
					learning_rat_etha=self.learning_rate_etha,
					momentum=self.momentum,
					add_bias=add_bias,
				)
				for _ in range(self.architecture[i])
			]
			self.NN.append(layer)

		for epoch in range(self.epochs):
			for i in range(len(self.inputs)):
				# online learning
				self.feedforward(input=self.inputs[i], i=i,out="output")
				error = self.expected_outputs[i] - self.output[i]
				self.backpropagation(error)
			error_sum = self.mean_squared_error(
				predictions=self.output, targets=self.expected_outputs
			)
			#error_sum=mean_squared_error(y_true=self.expected_outputs,y_pred= self.output, squared=False)
			print("epoch: {ep}, RMSE:{err}".format(ep=epoch + 1, err=error_sum))
			self.errors.append(np.sqrt(error_sum))
			#self.errors.append(error_sum)

			#validation pass for early stoping
			for i_val in range(len(self.inputs_val)):
				self.feedforward(input=self.inputs_val[i_val],i=i_val,out="output_val")
			error_val_sum = self.mean_squared_error(
			 	predictions=self.output_val, targets=self.expected_outputs_val
			)
			#error_val_sum=mean_squared_error(y_true=self.expected_outputs_val,y_pred= self.output_val, squared=False)
			#self.errors_val.append(error_val_sum)
			self.errors_val.append(np.sqrt(error_val_sum))
			print("VAL epoch: {ep}, RMSE:{err}".format(ep=epoch + 1, err=error_val_sum))

			#early stopping

			if self.early_s:
				stop= self.early_stopping(self.errors)
				if stop:
					print("Early stopping activated, delta value of: ",self.dif)
					break


		#final performance measures of the trained NN
		self.rmse=np.min(self.errors)
		self.rmse_val=np.min(self.errors_val)
		self.min_train=np.argmin(self.errors)
		self.min_val=np.argmin(self.errors_val)

	def plot(self)->None:
		"""plot the loss curves for training and validation,
		adding a line in the epoch where the validation has min MSE for early stoping
		"""

		plt.figure()
		plt.plot(self.errors,label="Train")
		plt.plot(self.errors_val,label="Validation")
		# Plot the vertical line at min_val
		plt.axvline(x=self.min_val, color='r', linestyle='--', label='Min RMSE in Validation')
		# Plot points at the intersection of the vertical line with the curves
		plt.scatter(self.min_val, self.errors[self.min_val], color='r', marker='o')
		plt.scatter(self.min_val, self.errors_val[self.min_val], color='r', marker='o')
		plt.title("Epoch vs RMSE")
		plt.xlabel("Epoch")
		plt.ylabel("Root Mean Squared Error")
		plt.legend()
		plt.show()


	def test(self, X:list, y:list)->list:
		"""Test the neural network on a separate dataset.

		Parameters
		----------
		X : list
			Input data for testing.
		y : list
			Target outputs for testing.

		Returns
		-------
		list
			Predicted outputs and the root mean squared error (RMSE) on the test data.
		"""
		X = [np.array(xi) for xi in X]
		y = [np.array(yi) for yi in y]
		output = []

		for i in range(len(X)):
			input = X[i]
			for layer in self.NN:
				new_input = [neuron.activate(input) for neuron in layer]
				input = new_input
			output.append(input)

		error_sum = np.sqrt(self.mean_squared_error(predictions=output, targets=y))
		return output, error_sum

	def predict(self,X:list)->list:
		"""Make predictions on new, unseen data.

		Parameters
		----------
		X : list
			Input data for prediction.

		Returns
		-------
		list
			Predicted outputs.
		"""
		X = [np.array(xi) for xi in X]
		output = []
		for i in range(len(X)):
			input = X[i]
			for layer in self.NN:
				new_input = [neuron.activate(input) for neuron in layer]
				input = new_input
			output.append(input)
		return output
