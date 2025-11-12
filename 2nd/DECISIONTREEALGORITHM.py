import math
from collections import Counter

# Sample dataset
dataset = [
['Sunny', 'Hot', 'High', 'Weak', 'No'],
['Sunny', 'Hot', 'High', 'Strong', 'No'],
['Overcast', 'Hot', 'High', 'Weak', 'Yes'],
['Rain', 'Mild', 'High', 'Weak', 'Yes'],
['Rain', 'Cool', 'Normal', 'Weak', 'Yes'],
['Rain', 'Cool', 'Normal', 'Strong', 'No'],
['Overcast', 'Cool', 'Normal', 'Strong', 'Yes'],
['Sunny', 'Mild', 'High', 'Weak', 'No'],
['Sunny', 'Cool', 'Normal', 'Weak', 'Yes'],
['Rain', 'Mild', 'Normal', 'Weak', 'Yes'],
['Sunny', 'Mild', 'Normal', 'Strong', 'Yes'],
['Overcast', 'Mild', 'High', 'Strong', 'Yes'],
['Overcast', 'Hot', 'Normal', 'Weak', 'Yes'],
['Rain', 'Mild', 'High', 'Strong', 'No'],
]

features = ['Outlook', 'Temperature', 'Humidity', 'Wind']

# Helper function to calculate entropy
def entropy(examples):
    labels = [example[-1] for example in examples]
    label_counts = Counter(labels)
    total = len(labels)
    if total == 0:
        return 0
    return -sum((count/total) * math.log2(count/total) for count in label_counts.values())

# Helper to split dataset
def split_dataset(dataset, feature_index, value):
    return [row for row in dataset if row[feature_index] == value]

# Information gain calculation
def info_gain(dataset, feature_index):
    total_entropy = entropy(dataset)
    values = set(row[feature_index] for row in dataset)
    weighted_entropy = 0.0
    for value in values:
        subset = split_dataset(dataset, feature_index, value)
        weighted_entropy += (len(subset)/len(dataset)) * entropy(subset)
    return total_entropy - weighted_entropy

# ID3 algorithm
def id3(dataset, features):
    labels = [row[-1] for row in dataset]
    
    # Base case: All labels are the same
    if labels.count(labels[0]) == len(labels):
        return labels[0]
    
    # Base case: No features left, return majority
    if not features:
        return Counter(labels).most_common(1)[0][0]
    
    # Find the best feature
    gains = [info_gain(dataset, i) for i in range(len(features))]
    best_index = gains.index(max(gains))
    best_feature = features[best_index]
    
    tree = {best_feature: {}}
    
    # Recurse on each value of the best feature
    feature_values = set(row[best_index] for row in dataset)
    
    remaining_features = features[:best_index] + features[best_index+1:]
    
    for value in feature_values:
        subset = split_dataset(dataset, best_index, value)
        
        # Prune the dataset for the recursive call
        subset_pruned = [row[:best_index] + row[best_index+1:] for row in subset]
        
        # Handle empty subset case (return majority label of parent)
        if not subset:
            subtree = Counter(labels).most_common(1)[0][0]
        else:
            subtree = id3(subset_pruned, remaining_features)
            
        tree[best_feature][value] = subtree
        
    return tree

# Prediction function
def classify(tree, sample, features):
    if not isinstance(tree, dict):
        return tree # Leaf node
    
    root = next(iter(tree))
    feature_index = features.index(root)
    sample_value = sample[feature_index]
    
    if sample_value not in tree[root]:
        # Value not seen during training,
        # In a real implementation, you might return a default or majority class.
        # Here we'll return None or a placeholder.
        return None # Or handle as per requirement
        
    subtree = tree[root][sample_value]
    
    # Create a copy of sample/features *without* the current root feature
    # This is necessary because the subtree was built on pruned data
    remaining_sample = sample[:feature_index] + sample[feature_index+1:]
    remaining_features = features[:feature_index] + features[feature_index+1:]
    
    return classify(subtree, remaining_sample, remaining_features)

# --- Execution ---

# Build the decision tree
decision_tree = id3(dataset, features)

# Pretty print the tree
import pprint
pprint.pprint(decision_tree)

# Classify a new sample
new_sample = ['Sunny', 'Cool', 'High', 'Strong'] # Should output 'No'
features_for_classification = ['Outlook', 'Temperature', 'Humidity', 'Wind'] # Original features list

prediction = classify(decision_tree, new_sample, features_for_classification)
print(f"\nPrediction for {new_sample} => PlayTennis: {prediction}")