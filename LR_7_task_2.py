# LR_7_task_2.py
import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
import matplotlib.pyplot as plt

# Завантажимо дані
input_file = 'data_regr_3.txt'
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

# Навчимо регресор
regressor.fit(X_train, y_train)

# Прогнозуємо вихідні значення для тестової вибірки
y_test_pred = regressor.predict(X_test)

# Побудуємо графік
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, y_test_pred, color='red', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear regression (variant 3)')
plt.show()

# Обчислимо метрики для оцінки якості регресії
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2))
print("Explained variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2))
print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))
