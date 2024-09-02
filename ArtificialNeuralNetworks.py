import pandas as pd
import tensorflow as tf
import numpy as np

dataset = pd.read_csv('Churn_Modelling.csv')
x = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x[:, 2] = le.fit_transform(x[:, 2])

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

# creating the neural network

ann = tf.keras.models.Sequential()

# hidden layers

ann.add(tf.keras.layers.Dense(units=12, activation='tanh'))
ann.add(tf.keras.layers.Dense(units=3, activation='relu'))
ann.add(tf.keras.layers.Dense(units=12, activation='tanh'))
ann.add(tf.keras.layers.Dense(units=3, activation='relu'))

# making the output layer

ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# training the dumb brain

ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
ann.fit(x_train, y_train, batch_size=5, epochs=100)

y_pred = ann.predict(x_test)
y_pred = (y_pred > 0.41)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
cm = confusion_matrix(y_test, y_pred)
print("\nthe confusion matrix\n", cm)
print("\nthe accuracy score\n", accuracy_score(y_test, y_pred))
