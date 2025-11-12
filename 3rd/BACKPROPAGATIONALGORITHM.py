import numpy as np
import math
from collections import Counter

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Training data for XOR
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])
y = np.array([[0],
              [1],
              [1],
              [0]])

input_layer_neurons = X.shape[1] 
hidden_layer_neurons = 2 
output_neurons = 1

np.random.seed(42) 
wh = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons))
bh = np.random.uniform(size=(1, hidden_layer_neurons))
wout = np.random.uniform(size=(hidden_layer_neurons, output_neurons))
bout = np.random.uniform(size=(1, output_neurons))

lr = 0.5 
epochs = 10000

for epoch in range(epochs):
    hidden_layer_input = np.dot(X, wh) + bh
    hidden_layer_output = sigmoid(hidden_layer_input)
    
    output_layer_input = np.dot(hidden_layer_output, wout) + bout
    predicted_output = sigmoid(output_layer_input)
    
    error = y - predicted_output
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    
    error_hidden_layer = d_predicted_output.dot(wout.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)
    
    wout += hidden_layer_output.T.dot(d_predicted_output) * lr
    bout += np.sum(d_predicted_output, axis=0, keepdims=True) * lr
    wh += X.T.dot(d_hidden_layer) * lr
    bh += np.sum(d_hidden_layer, axis=0, keepdims=True) * lr
    
    if epoch % 1000 == 0:
        loss = np.mean(np.square(error))
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

print("\nFinal predicted outputs after training:")
print(np.round(predicted_output, 3))

# Testing on new inputs
def predict(x):
    hidden_layer = sigmoid(np.dot(x, wh) + bh)
    output_layer = sigmoid(np.dot(hidden_layer, wout) + bout)
    return output_layer

test_samples = np.array([[0,0],[0,1],[1,0],[1,1]])
predictions = predict(test_samples)

print("\nTest Predictions (rounded):")
for inp, pred in zip(test_samples, predictions):
    print(f"Input: {inp} => Predicted Output: {np.round(pred[0])}")