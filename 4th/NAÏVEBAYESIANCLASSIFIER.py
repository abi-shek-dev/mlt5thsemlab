import csv
import math
from collections import defaultdict, Counter

def read_csv(filename):
    with open(filename, 'r') as f:
        data = list(csv.reader(f))
    header = data[0]
    rows = data[1:]
    X = [list(map(float, row[:-1])) for row in rows]
    y = [row[-1] for row in rows]
    return X, y

def mean(numbers):
    return sum(numbers) / float(len(numbers))

def variance(numbers, mean_val):
    if len(numbers) <= 1:
        return 0.0
    return sum([(x - mean_val) ** 2 for x in numbers]) / float(len(numbers) - 1)

def gaussian_pdf(x, mean, var):
    if var == 0:
        return 1.0 if x == mean else 0.0
    exponent = math.exp(-((x - mean) ** 2) / (2 * var))
    return (1 / math.sqrt(2 * math.pi * var)) * exponent

def train_naive_bayes(X, y):
    separated = defaultdict(list)
    for features, label in zip(X, y):
        separated[label].append(features)
    
    summaries = {}
    for class_value, instances in separated.items():
        summaries[class_value] = []
        for feature_column in zip(*instances):
            mean_val = mean(feature_column)
            var_val = variance(feature_column, mean_val)
            summaries[class_value].append((mean_val, var_val))
            
    return summaries

def calculate_class_probabilities(summaries, input_vector, class_priors):
    probabilities = {}
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = class_priors[class_value]
        for i in range(len(class_summaries)):
            mean_val, var_val = class_summaries[i]
            probabilities[class_value] *= gaussian_pdf(input_vector[i], mean_val, var_val)
    return probabilities

def predict(summaries, input_vector, class_priors):
    probabilities = calculate_class_probabilities(summaries, input_vector, class_priors)
    best_label, best_prob = None, -1
    for class_value, prob in probabilities.items():
        if best_label is None or prob > best_prob:
            best_prob = prob
            best_label = class_value
    return best_label

def accuracy_metric(actual, predicted):
    correct = sum(1 for a, p in zip(actual, predicted) if a == p)
    return correct / len(actual) * 100.0

def main():
    try:
        train_X, train_y = read_csv('4th/train.csv')
    except FileNotFoundError:
        print("Error: train.csv not found.")
        return
    except Exception as e:
        print(f"Error reading train.csv: {e}")
        return

    class_counts = Counter(train_y)
    total_count = len(train_y)
    class_priors = {cls: count / total_count for cls, count in class_counts.items()}

    summaries = train_naive_bayes(train_X, train_y)

    try:
        test_X, test_y = read_csv('4th/test.csv')
    except FileNotFoundError:
        print("Error: test.csv not found.")
        return
    except Exception as e:
        print(f"Error reading test.csv: {e}")
        return

    predictions = []
    for sample in test_X:
        pred = predict(summaries, sample, class_priors)
        predictions.append(pred)

    acc = accuracy_metric(test_y, predictions)
    print(f'Accuracy on test data: {acc:.2f}%')

if __name__ == "__main__":
    main()