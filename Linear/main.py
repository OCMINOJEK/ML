import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np

class SVM:
    def __init__(self, kernel='linear', C=1.0, gamma=0.1, degree=3, coeff=1, learning_rate=0.001,
                 max_iter=1000, tolerance=1e-4, random_state=0):
        self.kernel = kernel                # Тип ядра ('linear', 'polynomial', 'rbf')
        self.C = C                          # Коэффициент регуляризации
        self.gamma = gamma                  # Параметр ядра
        self.degree = degree                # Степень полинома
        self.coeff = coeff                  # Свободный коэффициент в полиномиальном ядре
        self.learning_rate = learning_rate  # Шаг градиентного спуска
        self.max_iter = max_iter            # Максимальное количество итераций
        self.tolerance = tolerance          # Допуск для проверки сходимости
        self.random_state = random_state    # Фиксация случайного состояния


    def _kernel_function(self, X, Z):
        if self.kernel == 'linear':
            return X @ Z.T
        elif self.kernel == 'polynomial':
            return (self.gamma * (X @ Z.T) + self.coeff) ** self.degree
        elif self.kernel == 'rbf':
            X_norm = np.sum(X**2, axis=1).reshape(-1, 1)
            Z_norm = np.sum(Z**2, axis=1).reshape(1, -1)
            return np.exp(-self.gamma * (X_norm + Z_norm - 2 * X @ Z.T))
        else:
            raise ValueError(f"Неизвестное ядро: {self.kernel}")


    def fit(self, X, y):
        np.random.seed(self.random_state)
        n, p = X.shape
        self.bias = 0
        self.weights = np.random.randn(p)

        K = self._kernel_function(X, X)

        for _ in range(self.max_iter):
            y_pred = K @ self.weights + self.bias
            margins = y * y_pred
            errors = margins < 1

            dw = np.zeros(p)
            db = 0

            for i in range(n):
                if errors[i]:
                    dw -= self.C * y[i] * X[i]
                    db -= self.C * y[i]

            self.weights -= self.learning_rate * (dw + self.weights)
            self.bias -= self.learning_rate * db

            if np.linalg.norm(dw) < self.tolerance and np.abs(db) < self.tolerance:
                break


    def predict(self, X):
        K = self._kernel_function(X, X)
        return np.sign(K @ self.weights + self.bias)


class GradientLinearClassifier:
    def __init__(self, learning_rate=0.001, tolerance=1e-5, alpha=0.5, lambda_=0, loss="logarithmic"):
        self.learning_rate = learning_rate  # Скорость обучения
        self.tolerance = tolerance          # Допустимая погрешность
        self.alpha = alpha                  # Коэффициент регуляризации
        self.lambda_ = lambda_              # Баланс между L1 и L2 регуляризацией (Elastic Net)
        self.loss = loss                    # Выбранная функция потерь ("logarithmic", "square", "exponential")


    def _margin(self, X, y):
        return y * (X @ self.weights)


    def _loss_and_gradient(self, X, y):
        n, p = X.shape
        margins = self._margin(X, y)

        # Выбор функции потерь
        if self.loss == "logarithmic":
            # LR
            # loss = np.log2(1 + np.exp(-margins)).mean()
            dL_dm = -y / (np.log(2) + np.exp(margins) * np.log(2))
        elif self.loss == "square":
            # LDA
            # loss = ((1 - margins)**2).mean()
            dL_dm = y * (2 * margins - 2)
        elif self.loss == "exponential":
            # AdaBoost
            # loss = np.exp(-margins).mean()
            dL_dm = -y * np.exp(-margins)
        else:
            raise ValueError("Неподдерживаемая функция потерь")

        dw = X.T @ dL_dm / n

        # Elastic Net регуляризация
        dw += self.alpha * (self.lambda_ * np.sign(self.weights) + (1 - self.lambda_) * self.weights)

        return dw


    def fit(self, X, y):
        n, p = X.shape
        self.weights = np.random.randn(p) * 0.01
        previous_dw = np.zeros(p)

        while True:
            dw = self._loss_and_gradient(X, y)
            self.weights = self.weights * (1 - self.learning_rate * self.lambda_) - self.learning_rate * dw

            dw_reduction = np.abs(dw - previous_dw)
            if np.all(dw_reduction < self.tolerance):
                break

            previous_dw = dw


    def predict(self, X):
        return np.sign(X @ self.weights)


class MatrixLinearClassifier:
    def __init__(self, alfa=0.5):
        self.alfa = alfa  # Коэффициент регуляризации


    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)  # добавляем вектор смещения
        n, p = X.shape  # p = 1 + признаки, n = количество данных
        I = np.eye(p)  # единичная матрица p * p
        I[0, 0] = 0  # не регуляризируем смещение
        XT_X_inv = np.linalg.inv(X.T @ X + self.alfa * I)  # (X.T * X + λI)^(-1)
        weights = np.linalg.multi_dot([XT_X_inv, X.T, y])  # (X.T * X + λI)^(-1) * X.T * y
        self.bias, self.weights = weights[0], weights[1:]


    def predict(self, X_test):
        linear_output = X_test @ self.weights + self.bias
        return np.sign(linear_output)


high_ranks = ['Претендент','Грандмастер']
other_ranks = ['Мастер', 'Алмаз I']
data = pd.read_csv("processed_data.csv")
data['Rank'] = data['Rank'].apply(lambda x: 1 if x in high_ranks else -1)

y = data['Rank']
X = data.drop(columns=['Rank'])
X = X.drop(X.columns[43:-1], axis=1)  # Удаляем favorite champions
X = X.drop(X.columns[37:-1], axis=1)  # Удаляем favorite roles

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifierGrad = GradientLinearClassifier()
classifierGrad.fit(X_train.values, y_train.values)
y_pred1 = classifierGrad.predict(X_test.values)
accuracy1 = accuracy_score(y_test, y_pred1)
print(f"Accuracy gradient: {accuracy1:.2f}")

classifierMatrix = MatrixLinearClassifier()
classifierMatrix.fit(X_train.values, y_train.values)
y_pred2 = classifierMatrix.predict(X_test.values)
accuracy2 = accuracy_score(y_test, y_pred2)
print(f"Accuracy matrix: {accuracy2:.2f}")