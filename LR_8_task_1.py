# LR_8_task_1.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Завантажимо набір даних
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

# Розділимо дані на навчальну та тестову вибірки
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.5, random_state = 0)

# Створимо модель лінійної регресії та навчимо її
regr = linear_model.LinearRegression()
regr.fit(Xtrain, ytrain)

# Зробимо прогноз по тестовій вибірці
ypred = regr.predict(Xtest)

# Розрахуємо та виведемо коефіцієнти регресії та показники якості
print("Коефіцієнти:", regr.coef_)
print("Вільний член:", regr.intercept_)
print("R2 =", round(r2_score(ytest, ypred), 2))
print("MAE =", round(mean_absolute_error(ytest, ypred), 2))
print("MSE =", round(mean_squared_error(ytest, ypred), 2))

# Побудуємо графік: виміряно vs передбачено
fig, ax = plt.subplots()
ax.scatter(ytest, ypred, edgecolors = (0, 0, 0))
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw = 4)
ax.set_xlabel('Виміряно')
ax.set_ylabel('Передбачено')
plt.show()
