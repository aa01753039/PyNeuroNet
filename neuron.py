"""
neuron.py

This module contains the implementation of the Neuron class, which represents a single neuron
in a neural network. The Neuron class includes methods for activation, backpropagation, and weight
updates, supporting both output and hidden layers.

Classes:
- Neuron: A class representing a single neuron in a neural network.

Example Usage:
```python
from neuron import Neuron

# Create a neuron with 3 input features, learning rate 0.1, momentum 0.2, and no bias
neuron = Neuron(n_input=3, learning_rate=0.1, momentum=0.2, add_bias=False)

# Activate the neuron with a list of inputs
output = neuron.activate([0.5, -0.2, 1.0])

# Backpropagate the error for the output layer
neuron.backpropagate_output(0.2)

# Backpropagate the error for a hidden layer with specified next layer gradient and weights
neuron.backpropagate_hidden(next_layer_gradient=[0.1, -0.3], next_layer_weights=[0.2, 0.5])

# Update the weights and bias of the neuron using gradient descent with momentum
neuron.update()

Author: Lesly Guerrrero
Date: October 24, 2023.

"""
import numpy as np
from numpy.random import uniform,seed

#seed used for make this a deterministic algorithm
seed(0)


class Neuron:
	def __init__(self, n_input:int, learning_rate_alpha:float,learning_rat_etha:float, momentum:float, add_bias:bool)->None:
		"""Initialize the Neuron with random weights, learning rate, momentum, and optional bias.

		Parameters
		----------
		n_input : int
			Number of input features.
		learning_rate_alpha : float
			The learning rate for activation function and gradient calculation.
		learning_rate_etha : float
			The learning rate for weight updates.
		momentum : float
			The momentum for weight updates.
		add_bias : bool
			Whether to include a bias in the neuron's computation.
		"""
		self.weights = uniform(low=-1, high=1, size=n_input) #array of size no. of input features of the previous layer, of floats between -1 and 1
		self.learning_rate_alpha = learning_rate_alpha #learning rate used in the feedforward and gradient
		self.learning_rat_etha = learning_rat_etha #learning rate used in the weight updates
		self.momentum = momentum
		self.prev_weight_update = np.zeros(n_input) #initialize the weight update as an array of zeros
		self.prev_bias_update = 0 #initialize the bias update as 0

		#decide wether bias is going to be activated or not
		if add_bias:
			self.bias = uniform(low=-1, high=1) #a float between -1 and 1
		else:
			self.bias = 0

	def sigmoide(self, x:float)->float:
		"""Sigmoid activation function.

		Parameters
		----------
		x : float
			Input to the sigmoid function.

		Returns
		-------
		float
			Output of the sigmoid function.
		"""
		return 1 / (1 + np.exp(-self.learning_rate_alpha * x)) #calculation of the sigmoid functionA

	def sigmoide_derivative(self, x:np.array)->float:
		"""Derivative of the sigmoid activation function.

		Parameters
		----------
		x : np.array
			 Output of the sigmoid function.

		Returns
		-------
		float
			Sigmoid derivative.
		"""
		return x * (1.0 - x) #derivate of the sigmoid function

	def weights_mult(self, x:np.array)->float:
		"""Calculate the weighted sum of inputs.

		Parameters
		----------
		x : np.array
			Input features.

		Returns
		-------
		float
			Weighted sum (dot product).
		"""
		return np.dot(x, self.weights) + self.bias #dot product between weights and bias (bias multiplication is ommited and is only added beacuse its input is 1, so technically its been multiplied times 1)

	def activate(self, inputs:list)->float:
		"""Activate the neuron using sigmoid as activation function and compute the output.

		Parameters
		----------
		inputs : list
			Input features.

		Returns
		-------
		float
			Neuron output.
		"""
		self.inputs = np.array(inputs) #for further calculations inputs need to be of type numpy array
		self.output = self.sigmoide(self.weights_mult(inputs)) #the ouput of the neuron is simply the sigmoid function on the weights multiplication of the inputs
		return self.output

	def backpropagate_ouput(self, error:float)->None:
		"""Backpropagate the error for output layer. (gradient)

		Parameters
		----------
		error : float
			Error signal.
		"""
		self.gradient = (
			self.learning_rate_alpha * self.sigmoide_derivative(self.output) * np.array(error)
		) #the ouput layer uses different formula for this calculation thats why it is in other function

	def backpropagate_hidden(self, next_layer_gradient:list, next_layer_weights:list)->None:
		"""Backpropagate the error for hidden layers. (gradient)

		Parameters
		----------
		next_layer_gradient : list
			Gradient from the next layer.
		next_layer_weights : list
			Weights of the next layer.
		"""
		self.gradient = (
			self.learning_rate_alpha
			* self.sigmoide_derivative(self.output)
			* np.dot(np.array(next_layer_gradient), np.array(next_layer_weights))
		)# this calculation fits to all the hidden layers in the neural network

	def update(self)->None:
		"""Update weights and bias using gradient descent with momentum.
		"""
		self.delta = self.learning_rat_etha * np.array(self.gradient) * self.inputs + (
			self.momentum * self.prev_weight_update
		)
		self.delta_bias = self.learning_rat_etha * self.gradient + (
			self.momentum * self.prev_bias_update # the delta for the bias is calculated appart from the wegths due to its input (1), it is never considered as a value in the vector of inputs therefore it isn't muliply by them
		)
		self.prev_weight_update = self.delta
		self.prev_bias_update = self.delta_bias #the bias is appart because it doesnt make part of the weight vector
		self.bias += self.delta_bias #it is a float so it can be added this way
		self.weights = np.add(self.weights, self.delta) #we need de numpy add since both of them (weights and delta) are vectors
