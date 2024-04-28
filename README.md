# Building-Neural-Network-using-only-NumPy-and-Mathematics
Implementation of neural network without the help of PyTorch or TensorFlow, using only NumPy and Mathematics.

## Objective 
Build a neural network using only NumPy and Mathematics.
## Methodology
### Data Collection
Data is imported directly from Keras. This is a dataset of 60,000 28x28 grayscale images of the 10 digits, along with a test set of 10,000 images.

### 1. Imports:

pandas as pd: Not used in this specific script, but likely included for potential future data manipulation.
numpy as np: Provides mathematical functions and array operations essential for neural networks.
pickle: Used for saving and loading the trained model parameters.
from keras.datasets import mnist: Imports the MNIST dataset containing handwritten digits.
matplotlib.pyplot as plt: Used for visualization (plotting images in this case).

### 2. Activation Functions:

ReLU(Z): Implements the ReLU (Rectified Linear Unit) activation function. It sets any negative value in Z to zero and keeps positive values unchanged.
derivative_ReLU(Z): Calculates the derivative of the ReLU function, used in backpropagation.
softmax(Z): Implements the softmax function, which normalizes the output of a layer to probabilities between 0 and 1.

### 3. Initialization:

init_params(): Initializes the weights (W1, W2) and biases (b1, b2) of the neural network. These are randomly generated with Xavier initialization (using square root of fan-in) for better convergence.

### 4. Forward Propagation:

forward_propagation(X, W1, b1, W2, b2): Performs a single forward pass through the network. It takes the input data (X), weights, and biases as arguments and calculates the activations for each layer (Z1, A1, Z2, A2).

### 5. One-Hot Encoding:

one_hot(Y): Converts the integer class labels (Y) in the training data to one-hot encoded vectors. This is a common practice for representing categorical data in neural networks.

### 6. Backpropagation:

backward_propagation(X, Y, A1, A2, W2, Z1, m): Implements backpropagation to calculate the gradients of the loss function with respect to the weights and biases. It uses the chain rule to calculate the gradients for each layer.

### 7. Weight Update:

update_params(alpha, W1, b1, W2, b2, dW1, db1, dW2, db2): Updates the weights and biases based on the calculated gradients (dW1, db1, dW2, db2) and a learning rate (alpha).

### 8. Predictions and Accuracy:

get_predictions(A2): Gets the predicted class labels for the input data by finding the index of the maximum value in the output activation (A2).
get_accuracy(predictions, Y): Calculates the accuracy of the model by comparing the predicted labels with the true labels (Y).

### 9. Gradient Descent:

gradient_descent(X, Y, alpha, iterations): Trains the neural network using gradient descent. It iterates for a specified number of epochs (iterations) performing forward propagation, backpropagation, and weight updates. It also prints the training accuracy at regular intervals.

### 10. Making Predictions:

make_predictions(X, W1 ,b1, W2, b2): Given a new input data (X), performs a forward pass through the trained network using the stored weights and biases to predict the class label.

### 11. Visualization:

show_prediction(index,X, Y, W1, b1, W2, b2): Shows a specific image from the test data (X) along with its predicted and actual labels.

### 12. Main Section:

Loads the MNIST dataset using mnist.load_data.
Preprocesses the data by reshaping and normalizing the image pixels.
Trains the network using gradient_descent.
Saves the trained model parameters using pickle.
Loads the trained parameters.
Makes predictions on a few test images and visualizes them using show_prediction.

## Results
### Summary Statistics
Accuracy = 92.000%
![Accuracy](MNIST/Results/Screenshot 2024-04-29 at 1.04.23 AM)
