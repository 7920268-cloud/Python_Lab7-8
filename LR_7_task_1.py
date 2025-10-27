# LR_7_task_1.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import sklearn.metrics as sm
import pickle

# 1. Завантаження даних
input_file = 'data_singlevar_regr.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# 2. Розбиття на навчальну та тестову вибірки (80/20)
num_training = int(0.8 * len(X))
X_train, y_train = X[:num_training], y[:num_training]
X_test, y_test = X[num_training:], y[num_training:]

# 3. Навчання моделі
regressor = linear_model.LinearRegression()
regressor.fit(X_train, y_train)

# 4. Прогноз
y_pred = regressor.predict(X_test)

# 5. Графік
plt.scatter(X_test, y_test, color='green')
plt.plot(X_test, y_pred, color='black', linewidth=2)
plt.title('Linear Regression (Single Variable)')
plt.xlabel('X')
plt.ylabel('y')
plt.savefig('lr7_task1_plot.png')
plt.show()

# 6. Метрики
print("MAE =", round(sm.mean_absolute_error(y_test, y_pred), 3))
print("MSE =", round(sm.mean_squared_error(y_test, y_pred), 3))
print("R2  =", round(sm.r2_score(y_test, y_pred), 3))

# 7. Збереження моделі
with open('model_lr7_task1.pkl', 'rb') as f:
    regressor_model = pickle.load(f)
y_test_pred_new = regressor_model.predict (X_test)
print ("\nNew mean absolute error =", round (sm.mean_absolute_error(y_test, y_test_pred_new), 2))
