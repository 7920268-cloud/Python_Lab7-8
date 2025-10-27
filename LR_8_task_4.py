# LR_8_task_4.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# Завантажимо набір даних
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

# Розділимо дані
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=0
)

# Створимо модель
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)

# Прогноз
y_pred = regr.predict(X_test)

# Метрики
print("Коефіцієнти:", regr.coef_)
print("Вільний член:", regr.intercept_)
print("R2 =", round(r2_score(y_test, y_pred), 2))
print("MAE =", round(mean_absolute_error(y_test, y_pred), 2))
print("MSE =", round(mean_squared_error(y_test, y_pred), 2))

# Побудуємо графік
plt.scatter(y_test, y_pred, color='blue', edgecolors='black')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.title('Diabetes dataset regression')
plt.show()
