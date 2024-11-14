







import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import numpy as np

class GradientLinearClassifier:
    def __init__(self, learning_rate=0.001, tolerance=1e-4, alpha=0.5, lambda_=0, loss="logarithmic"):
        self.learning_rate = learning_rate  # Скорость обучения
        self.tolerance = tolerance          # Допустимая погрешность
        self.alpha = alpha                  # Коэффициент регуляризации
        self.lambda_ = lambda_              # Баланс между L1 и L2 регуляризацией (Elastic Net)
        self.loss = loss                    # Выбранная функция потерь ("logarithmic", "square", "exponential")

    def _margin(self, X, y):
        """Вычисление отступа m = y * (Xw + b)."""
        return y * (X @ self.weights + self.bias)

    def _loss_and_gradient(self, X, y):
        """Вычисляем функцию потерь и градиенты в зависимости от выбранного эмпирического риска."""
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

        db = dL_dm.mean()
        dw = X.T @ dL_dm / n

        # Elastic Net регуляризация
        dw += self.alpha * (self.lambda_ * np.sign(self.weights) + (1 - self.lambda_) * self.weights)

        return db, dw

    def fit(self, X, y):
        n, p = X.shape
        self.bias, self.weights = 0, np.zeros(p)
        previous_db, previous_dw = 0, np.zeros(p)

        while True:
            db, dw = self._loss_and_gradient(X, y)

            self.bias -=  self.learning_rate * db
            self.weights = self.weights * (1 - self.learning_rate * self.lambda_) - self.learning_rate * dw

            db_reduction = np.abs(db - previous_db)
            dw_reduction = np.abs(dw - previous_dw)
            if db_reduction < self.tolerance and dw_reduction.all() < self.tolerance:
                break

            previous_db = db
            previous_dw = dw

    def predict(self, X):
        # Предсказываем метки класса (±1)
        return np.sign(X @ self.weights + self.bias)


class MatrixLinearClassifier:
    def __init__(self, lambda_=0.0):
        self.lambda_ = lambda_

    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)  # добавляем вектор смещения
        n, p = X.shape  # p = 1 + признаки, n = количество данных
        I = np.eye(p)  # единичная матрица p * p
        I[0, 0] = 0  # не регуляризируем смещение
        XT_X_inv = np.linalg.inv(X.T @ X + self.lambda_ * I)  # (X.T * X + λI)^(-1)
        weights = np.linalg.multi_dot([XT_X_inv, X.T, y])  # (X.T * X + λI)^(-1) * X.T * y
        self.bias, self.weights = weights[0], weights[1:]

    def predict(self, X_test):
        linear_output = X_test @ self.weights + self.bias
        return np.sign(linear_output)  # возвращаем метки классов ±1

high_ranks = ['Претендент','Грандмастер']
other_ranks = ['Мастер', 'Алмаз I']
data = pd.read_csv("processed_data.csv")
data['Rank'] = data['Rank'].apply(lambda x: 1 if x in high_ranks else -1)

y = data['Rank']
X = data.drop(columns=['Rank'])
X = X.drop(X.columns[43:-1], axis=1)  # Удаляем favorite champions
X = X.drop(X.columns[37:-1], axis=1)  # Удаляем favorite roles

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifier = GradientLinearClassifier()
classifier.fit(X_train.values, y_train.values)

# Предсказание и оценка
y_pred = classifier.predict(X_test.values)

# Оценка точности
accuracy = np.mean(y_pred == y_test.values)
print(f"Accuracy: {accuracy:.2f}")
