from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

results = []
correct = 0
incorrect = 0

print("\nPrediction Results:")
print(f"{'Index':<6} {'Predicted':<15} {'Actual':<15} {'Status'}")
print("-" * 50)

for i in range(len(y_test)):
    pred = target_names[y_pred[i]]
    actual = target_names[y_test[i]]
    status = "Correct" if y_pred[i] == y_test[i] else "Wrong"
    
    if status == "Correct":
        correct += 1
    else:
        incorrect += 1
        
    print(f"{i:<6} {pred:<15} {actual:<15} {status}")

print("\nSummary:")
print(f"Total samples: {len(y_test)}")
print(f"Correct predictions: {correct}")
print(f"Wrong predictions: {incorrect}")
print(f"Accuracy: {correct / len(y_test):.2f}")