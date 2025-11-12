import csv

def load_data(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        headers = next(reader)
        data = [row for row in reader]
    return headers, data

def is_consistent(hypothesis, example):
    return all(h == '?' or h == e for h, e in zip(hypothesis, example))

def generalize(hypothesis, example):
    return ['?' if h != e else h for h, e in zip(hypothesis, example)]

def candidate_elimination(data):
    num_features = len(data[0]) - 1

    for row in data:
        if row[-1] == 'Yes':
            S = row[:-1]
            break
    else:
        raise ValueError("No positive examples found!")

    G = [['?'] * num_features]

    for example in data:
        inputs, label = example[:-1], example[-1]
        if label == 'Yes':
            G = [g for g in G if is_consistent(g, inputs)]
            S = generalize(S, inputs)
        else:
            G_new = []
            for g in G:
                if is_consistent(g, inputs):
                    for i in range(num_features):
                        if g[i] == '?':
                            values = set(x[i] for x in data if x[-1] == 'Yes')
                            for value in values:
                                if value != inputs[i]:
                                    new_hyp = g.copy()
                                    new_hyp[i] = value
                                    if any(is_consistent(new_hyp, x[:-1]) for x in data if x[-1] == 'Yes'):
                                        G_new.append(new_hyp)
                else:
                    G_new.append(g)
            G = G_new

    G_final = []
    for g in G:
        if not any(other != g and all((gc == '?' or gc == oc) for gc, oc in zip(g, other)) for other in G):
            G_final.append(g)

    return S, G_final

filename = '1st/training_data.csv'
headers, data = load_data(filename)
S_final, G_final = candidate_elimination(data)

print("\n✅ Final Specific Hypothesis (S):")
print(S_final)

print("\n✅ Final General Hypotheses (G):")
if not G_final:
    print("No consistent general hypotheses found.")
else:
    for g in G_final:
        print(g)