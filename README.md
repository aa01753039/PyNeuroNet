# PyNeuroNet

Welcome to the **Neural Network Implementation** repository! This project provides a custom implementation of a neural network, including neuron-level computations and features such as backpropagation, weight updates, early stopping, and visualization of training performance.

## Features
- Custom **Neuron** class with sigmoid activation and gradient-based weight updates.
- **NeuralNetwork** class for creating, training, testing, and predicting using feedforward neural networks.
- Support for multiple hidden layers and early stopping.
- Integration with **scikit-learn** for data scaling and splitting.
- Visualization of training and validation performance curves.

---

## File Structure
### `neuron.py`
Contains the `Neuron` class, representing individual neurons:
- Implements sigmoid activation, backpropagation, and momentum-based weight updates.
- Handles computations for both hidden and output layers.

### `neural_network.py`
Contains the `NeuralNetwork` class, which:
- Creates network architecture.
- Trains using backpropagation with momentum and learning rate adjustments.
- Supports early stopping and tracks RMSE for training and validation.

### `main.py`
Script to:
- Load and preprocess data.
- Create and train the neural network.
- Visualize training results and save the trained model.

### `NeuralNetHolder`
A utility class for loading trained models and scalers for predictions.

---

## Getting Started

### Prerequisites
- Python 3.8+
- Required libraries:
  - `numpy`
  - `matplotlib`
  - `scikit-learn`
  - `pandas`

Install dependencies with:
```bash
pip install -r requirements.txt
```

### Usage

#### Training the Model
1. Place your dataset (`ce889_dataCollection.csv`) in the project directory.
2. Run `main.py` to train the neural network:
   ```bash
   python main.py
   ```
3. Adjust hyperparameters and architecture in the script as needed.

#### Testing and Predictions
- Use the `test` method in `NeuralNetwork` to evaluate the trained model.
- Utilize `NeuralNetHolder` to make predictions on new data.

#### Example Code
```python
from neural_network import NeuralNetwork

# Initialize the neural network
nn = NeuralNetwork(learning_rate_alpha=0.5, learning_rate_etha=0.01, momentum=0.9)

# Train on data
nn.fit(x_train, y_train, x_val, y_val, hidden_layer=[10, 5], n_epochs=100)

# Test the model
predictions, rmse = nn.test(x_test, y_test)
print(f"RMSE on test set: {rmse}")

# Predict on new data
new_data = [[0.5, 0.3]]
predicted = nn.predict(new_data)
print(f"Predicted values: {predicted}")
```

#### Visualization
After training, view RMSE curves:
```python
nn.plot()
```

---

## Results
- Displays RMSE for training, validation, and testing.
- Supports saving the trained model and scalers using `pickle`.

---

## Contributing
Contributions are welcome! Feel free to submit issues or pull requests.

---

## Author
Lesly Guerrero

---

## License
This project is licensed under the MIT License.
