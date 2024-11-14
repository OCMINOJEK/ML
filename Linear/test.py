import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from numpy.linalg import norm

import numpy as np



class SVM:
    def __init__(self, kernel='poly', C=10.0, max_iter=1000, degree=3, gamma=1):
        self.kernel_type = kernel
        self.C = C
        self.max_iter = max_iter
        self.degree = degree
        self.gamma = gamma

    # Метод для вычисления ядра
    def kernel(self, x, y):
        if self.kernel_type == 'linear':
            return np.dot(x, y.T)
        elif self.kernel_type == 'poly':
            return np.dot(x, y.T) ** self.degree
        elif self.kernel_type == 'rbf':
            return np.exp(-self.gamma * np.sum((y - x[:, np.newaxis]) ** 2, axis=-1))
        else:
            raise ValueError("Неизвестное ядро")

    def fit(self, X, y):
        self.X_train = X
        self.labels = y
        self.alpha = np.zeros_like(self.labels, dtype=float)
        self.b = 0
        kernel_matrix = self.kernel(self.X_train, self.X_train) * self.labels.reshape(-1, 1) * self.labels

        n_samples = len(self.labels)
        iterations = 0
        while iterations < self.max_iter:
            alpha_pairs_changed = 0
            for i in range(n_samples):
                Ei = np.dot(self.alpha * self.labels, kernel_matrix[i]) + self.b - self.labels[i]
                if (self.labels[i] * Ei < -1e-3 and self.alpha[i] < self.C) or (
                        self.labels[i] * Ei > 1e-3 and self.alpha[i] > 0):
                    j = np.random.randint(0, n_samples)
                    while j == i:
                        j = np.random.randint(0, n_samples)

                    Ej = np.dot(self.alpha * self.labels, kernel_matrix[:, j]) + self.b - self.labels[j]

                    alpha_i_old = self.alpha[i]
                    alpha_j_old = self.alpha[j]

                    if self.labels[i] != self.labels[j]:
                        L = max(0, self.alpha[j] - self.alpha[i])
                        H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
                    else:
                        L = max(0, self.alpha[i] + self.alpha[j] - self.C)
                        H = min(self.C, self.alpha[i] + self.alpha[j])

                    if L == H:
                        continue

                    eta = 2 * kernel_matrix[i, j] - kernel_matrix[i, i] - kernel_matrix[j, j]
                    if eta >= 0:
                        continue

                    self.alpha[j] -= (self.labels[j] * (Ei - Ej)) / eta
                    self.alpha[j] = np.clip(self.alpha[j], L, H)

                    if np.abs(self.alpha[j] - alpha_j_old) < 1e-5:
                        continue

                    self.alpha[i] += self.labels[i] * self.labels[j] * (alpha_j_old - self.alpha[j])

                    b1 = self.b - Ei - self.labels[i] * (self.alpha[i] - alpha_i_old) * kernel_matrix[i, i] - \
                         self.labels[j] * (self.alpha[j] - alpha_j_old) * kernel_matrix[i, j]
                    b2 = self.b - Ej - self.labels[i] * (self.alpha[i] - alpha_i_old) * kernel_matrix[i, j] - \
                         self.labels[j] * (self.alpha[j] - alpha_j_old) * kernel_matrix[j, j]

                    if 0 < self.alpha[i] < self.C:
                        self.b = b1
                    elif 0 < self.alpha[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2

                    alpha_pairs_changed += 1

            iterations += 1
            if alpha_pairs_changed == 0:
                break

        # Определяем индексы опорных векторов
        support_indices = np.where(self.alpha > 1E-15)[0]
        # Вычисляем свободный член b
        self.b = np.mean((1 - np.sum(kernel_matrix[support_indices] * self.alpha, axis=1)) * self.labels[support_indices])

    def decision_function(self, X):
        kernel_eval = self.kernel(X, self.X_train)
        return np.dot(kernel_eval * self.labels, self.alpha) + self.b

    def predict(self, X):
        # Предсказываем метки классов
        predictions = np.sign(self.decision_function(X))
        return (predictions + 1) // 2


class GradientLinearClassifier:
    def __init__(self, learning_rate=0.001, tolerance=1e-5, alpha=0.5, lambda_=0, loss="logarithmic"):
        self.learning_rate = learning_rate  # Скорость обучения
        self.tolerance = tolerance  # Допустимая погрешность
        self.alpha = alpha  # Коэффициент регуляризации
        self.lambda_ = lambda_  # Баланс между L1 и L2 регуляризацией (Elastic Net)
        self.loss = loss  # Выбранная функция потерь ("logarithmic", "square", "exponential")

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


high_ranks = ['Претендент', 'Грандмастер']
other_ranks = ['Мастер', 'Алмаз I']
data = pd.read_csv("processed_data.csv")
data['Rank'] = data['Rank'].apply(lambda x: 1 if x in high_ranks else -1)

y = data['Rank']
X = data.drop(columns=['Rank'])
X = X.drop(X.columns[43:-1], axis=1)  # Удаляем favorite champions
X = X.drop(X.columns[37:-1], axis=1)  # Удаляем favorite roles

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=42)

# classifierGrad = GradientLinearClassifier()
# classifierGrad.fit(X_train.values, y_train.values)
# y_pred1 = classifierGrad.predict(X_test.values)
# accuracy1 = accuracy_score(y_test, y_pred1)
# print(f"Accuracy gradient: {accuracy1:.2f}")
#
# classifierMatrix = MatrixLinearClassifier()
# classifierMatrix.fit(X_train.values, y_train.values)
# y_pred2 = classifierMatrix.predict(X_test.values)
# accuracy2 = accuracy_score(y_test, y_pred2)
# print(f"Accuracy matrix: {accuracy2:.2f}")

# classifierSVM = SVM()
# classifierSVM.fit(X_train.values, y_train.values)
# y_pred3 = classifierSVM.predict(X_test.values)
# accuracy3 = accuracy_score(y_test, y_pred3)
# print(f"Accuracy SVM: {accuracy3:.2f}")