import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Data1.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x)
x = imputer.transform(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()

x_train = sc_x.fit_transform(x_train)
y_train = y_train.reshape(-1, 1)
y_train = sc_y.fit_transform(y_train).flatten()

from sklearn.svm import SVR
re = SVR(kernel='rbf')
re.fit(x_train, y_train)

y_predicted = sc_y.inverse_transform(re.predict(sc_x.transform(x_test)).reshape(-1, 1))

plt.scatter(y_test, y_predicted, color='red')
plt.plot(y_test, y_test, color='blue')
plt.title('Support Vector Regression (Predicted vs Actual)')
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.show()

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
print("R² Score:", r2_score(y_test, y_predicted))
print("Mean Squared Error:", mean_squared_error(y_test, y_predicted))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_predicted))
