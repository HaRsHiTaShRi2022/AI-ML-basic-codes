import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataset = pd.read_csv('Wine.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

from sklearn.decomposition import KernelPCA
lda = KernelPCA(n_components=2, kernel='rbf')
x_train = lda.fit_transform(x_train)
x_test = lda.transform(x_test)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train, y_train)

y_predicted = lr.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predicted)

print(cm)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_predicted)

print(ac)
