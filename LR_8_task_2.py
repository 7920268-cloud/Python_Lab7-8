# LR_8_task_2.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Згенеруємо дані
m = 100
X = 6 * np.random.rand(m, 1) - 8
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)

# Побудуємо графік
plt.scatter(X, y, color='blue')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Generated data (variant 8)')
plt.show()

# Створимо поліноміальні ознаки
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

# Побудуємо модель поліноміальної регресії
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)

# Прогноз
X_new = np.linspace(-8, 8, 100).reshape(100, 1)
X_new_poly = poly_features.transform(X_new)
y_new = poly_reg.predict(X_new_poly)

# Побудуємо графік
plt.scatter(X, y, color='blue')
plt.plot(X_new, y_new, color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Polynomial regression (variant 8)')
plt.show()

# Виведемо коефіцієнти
print('Коефіцієнти:', poly_reg.coef_)
print('Вільний член:', poly_reg.intercept_)
