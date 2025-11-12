import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
X = np.linspace(-3, 3, 100)
y = np.sin(X) + np.random.normal(0, 0.1, len(X))

X = X[:, np.newaxis]

def add_bias(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])

def locally_weighted_regression(X_train, y_train, x_query, tau=0.5):
    m = X_train.shape[0]
    W = np.eye(m)
    
    for i in range(m):
        diff = x_query - X_train[i]
        W[i, i] = np.exp(-(diff @ diff.T) / (2 * tau ** 2))
        
    X_bias = add_bias(X_train)
    x_query_bias = np.array([1, x_query[0]]) # Corrected for single query
    
    try:
        theta = np.linalg.pinv(X_bias.T @ W @ X_bias) @ X_bias.T @ W @ y_train
    except np.linalg.LinAlgError:
        return 0 
        
    return x_query_bias @ theta

X_plot = np.linspace(-3, 3, 300)[:, np.newaxis]
y_pred = np.array([
    locally_weighted_regression(X, y, x_query, tau=0.3)
    for x_query in X_plot
])

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='red', label='Noisy Data')
plt.plot(X_plot, y_pred.flatten(), color='blue', label='LWR Fit (tau=0.3)', linewidth=2) # Flatten y_pred
plt.title("Locally Weighted Regression (Non-Parametric)")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()