import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)
data = []
for i in range(0, len(dataset)):
    data.append([str(dataset.values[i, j]) for j in range(0, 20)])

from apyori import apriori
rules = apriori(transactions=data, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2, max_length=2)
results = list(rules)

print(results)

def inspect(results):
    lhs = [tuple(result[2][0][0])[0] for result in results]
    rhs = [tuple(result[2][0][1])[0] for result in results]
    supports = [result[1] for result in results]
    confidence = [result[2][0][2] for result in results]
    lifts = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidence, lifts))

resultinData = pd.DataFrame(inspect(results),columns=['Left hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])
print(resultinData)
