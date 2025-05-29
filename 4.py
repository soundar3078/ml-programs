#program4

import pandas as pd

def read_csv(file_path):
    data = pd.read_csv(file_path)
    print("Training data:")
    print(data)
    return data

def find_S_algorithm(data):
    hypothesis = None
    for i in range(len(data)):
        if data.iloc[i, -1].lower() == 'yes':
            hypothesis = data.iloc[i, :-1].values.tolist()
            break

    if hypothesis is None:
        raise ValueError("No positive example found in the dataset.")

    for i in range(len(data)):
        if data.iloc[i, -1].lower() == 'yes':
            for j in range(len(hypothesis)):
                if hypothesis[j] != data.iloc[i, j]:
                    hypothesis[j] = "?"

    return hypothesis

file_path = 'enjoysport.csv'
data = read_csv(file_path)
hypothesis = find_S_algorithm(data)
print("\nFinal hypothesis:", hypothesis)
