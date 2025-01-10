import pickle
class NeuralNetHolder:

	def __init__(self):
		#trained neural network
		with open("red_entrenada.pickle","rb") as file:
			self.nn = pickle.load(file)
		#MinMax scaler X
		with open("scalador_x.pickle","rb") as file:
			self.Xscale = pickle.load(file)
		#MinMax scaler Y
		with open("scalador_y.pickle","rb") as file:
			self.Yscale = pickle.load(file)


	def predict(self, input_row):
		# WRITE CODE TO PROCESS INPUT ROW AND PREDICT X_Velocity and Y_Velocity
		# Split the string into two substrings using the comma as a delimiter
		numbers_as_strings = input_row.split(',')
		# Convert the strings to float numbers
		input = [[float(num) for num in numbers_as_strings]]
		scaled_input=self.Xscale.transform(input)
		output=self.nn.predict(scaled_input)
		descaled=self.Yscale.inverse_transform(output)

		return descaled[0]

