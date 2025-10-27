# LR_7_task_3.py
import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
from sklearn.preprocessing import PolynomialFeatures

# Завантажимо дані
input_file = 'data_multivar_regr.txt'
data = np.loadtxt(input_file, delimiter=',')

# Розділимо дані на вхідні і вихідні змінні
X, y = data[:, :-1], data[:, -1]

# Розділимо дані на навчальну і тестову вибірки
num_training = int(0.8 * len(X))
num_test = len(X) - num_training

# Навчальні дані
X_train, y_train = X[:num_training], y[:num_training]

# Тестові дані
X_test, y_test = X[num_training:], y[num_training:]

# Створимо модель лінійної регресії
regressor = linear_model.LinearRegression()
regressor.fit(X_train, y_train)

# Прогноз для тестових даних
y_test_pred = regressor.predict(X_test)

# Виведемо результати
print("\nLinear regression performance:")
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))

# Створимо поліноміальні ознаки (ступінь 10)
polynomial = PolynomialFeatures(degree=10)
X_train_transformed = polynomial.fit_transform(X_train)
X_test_transformed = polynomial.transform(X_test)

# Поліноміальна регресія
poly_regressor = linear_model.LinearRegression()
poly_regressor.fit(X_train_transformed, y_train)

# Прогноз для тестових даних
y_test_pred_poly = poly_regressor.predict(X_test_transformed)

# Виведемо результати
print("\nPolynomial regression performance:")
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred_poly), 2))
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred_poly), 2))
print("R2 score =", round(sm.r2_score(y_test, y_test_pred_poly), 2))

# Передбачення для довільної точки
datapoint = [[7.75, 6.35, 5.56]]
poly_datapoint = polynomial.transform(datapoint)
print("\nLinear regression prediction:", regressor.predict(datapoint))
print("Polynomial regression prediction:", poly_regressor.predict(poly_datapoint))
