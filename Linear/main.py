import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from numpy.linalg import norm
import random
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


class SVM:
    def __init__(self, C=100.0, max_iter=100, kernel='linear', learning_rate=0.001, gamma=1,degree=5):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.kernel_type = kernel
        self.C = C
        self.degree = degree
        self.gamma = gamma


    def fit(self, x, y):
        self.X, self.y = x, y
        self.b = 0
        n, p = self.X.shape
        self.alpha = np.zeros(n)

        kernel_matrix = self.kernel(self.X, self.X)
        for i in range(self.max_iter):
            for j in range(n):
                self.alpha[j] -= self.learning_rate * (np.sum(self.alpha * y * kernel_matrix.T[i]) - 1)
                self.alpha[j] = max(0, min(self.alpha[j], self.C)) # 0 <= alpha[i] <= C

        diff = y - np.dot(kernel_matrix, (self.alpha * y))
        self.b = sum(diff)/ len(diff)


    def kernel(self, x, y):
        if self.kernel_type == 'linear':
            return np.dot(x, y.T)
        elif self.kernel_type == 'poly':
            return np.dot(x, y.T) ** self.degree
        elif self.kernel_type == 'rbf':
            return np.exp(-self.gamma * np.sum((y - x[:, np.newaxis]) ** 2, axis=-1))
        else:
            raise ValueError("Неизвестное ядро")

    def predict(self, x):
        return np.sign(np.dot(self.kernel(x, self.X), (self.alpha * self.y)) + self.b)



class GradientLinearClassifier:
    def __init__(self, learning_rate=0.001, tolerance=1e-5, alpha=0.5, lambda_=0, loss="logarithmic", max_iter=1000):
        self.learning_rate = learning_rate  # Скорость обучения
        self.tolerance = tolerance  # Допустимая погрешность
        self.alpha = alpha  # Коэффициент регуляризации
        self.lambda_ = lambda_  # Баланс между L1 и L2 регуляризацией (Elastic Net)
        self.loss = loss  # Выбранная функция потерь ("logarithmic", "square", "exponential")
        self.max_iter = max_iter

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
        iterations = 0

        while iterations < self.max_iter:
            dw = self._loss_and_gradient(X, y)
            self.weights = self.weights * (1 - self.learning_rate * self.lambda_) - self.learning_rate * dw

            dw_reduction = np.abs(dw - previous_dw)
            if np.all(dw_reduction < self.tolerance):
                break

            previous_dw = dw
            iterations += 1

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)

def plot_learning_curve(classifier, X, y):

    train_sizes = np.linspace(0.1, 0.9, 9)
    train_scores = []
    val_scores = []

    for train_size in train_sizes:
        X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=train_size, random_state=42)

        classifier.fit(X_train.values, y_train.values)
        train_score = accuracy_score(y_train, classifier.predict(X_train.values))
        val_score = accuracy_score(y_val, classifier.predict(X_val.values))

        train_scores.append(train_score)
        val_scores.append(val_score)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_scores, label='Training Accuracy')
    plt.plot(train_sizes, val_scores, label='Validation Accuracy')
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy')
    plt.title('Learning Curve')
    plt.legend()
    plt.show()

def plot_test_error_curve(classifier, X, y, n_splits=5):
    train_sizes = np.linspace(0.1, 0.9, 10)  # Изменили диапазон, чтобы исключить 1.0
    mean_errors = []
    ci_lower = []
    ci_upper = []

    for size in train_sizes:
        test_errors = []

        for i in range(n_splits):
            # Проверяем, чтобы размер тестового множества был больше 0
            if size >= 1.0:
                continue

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - size, random_state=None)

            classifier.fit(X_train.values, y_train.values)
            y_pred_test = classifier.predict(X_test.values)
            error = 1 - accuracy_score(y_test, y_pred_test)
            test_errors.append(error)
            print(size, i)
        mean_errors.append(np.mean(test_errors))
        ci_lower.append(np.percentile(test_errors, 2.5))
        ci_upper.append(np.percentile(test_errors, 97.5))

    plt.plot(train_sizes, mean_errors, label='Test Error', color='red')
    plt.fill_between(train_sizes, ci_lower, ci_upper, color='red', alpha=0.2, label='95% Confidence Interval')

    plt.xlabel("Training Set Size")
    plt.ylabel("Test Error")
    plt.title("Test Error Curve with Confidence Interval")
    plt.legend()
    plt.show()

def plot_test_error_curve_with_baseline(classifier, X, y, n_splits=5):
    train_sizes = np.linspace(0.1, 0.9, 10)
    mean_errors = []
    ci_lower = []
    ci_upper = []

    # Вычисление ошибки для различных размеров обучающего набора
    for size in train_sizes:
        test_errors = []

        for _ in range(n_splits):
            if size >= 1.0:
                continue

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - size, random_state=None)

            # Обучаем и тестируем классификатор (например, SVM)
            classifier.fit(X_train, y_train)
            y_pred_test = classifier.predict(X_test)
            error = 1 - accuracy_score(y_test, y_pred_test)
            test_errors.append(error)

        mean_errors.append(np.mean(test_errors))
        ci_lower.append(np.percentile(test_errors, 2.5))
        ci_upper.append(np.percentile(test_errors, 97.5))

    # Оцениваем ошибку на тестовом множестве для базового классификатора
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    classifier.fit(X_train, y_train)
    baseline_y_pred = classifier.predict(X_test)
    baseline_error = 1 - accuracy_score(y_test, baseline_y_pred)

    # Построение графика
    plt.plot(train_sizes, mean_errors, label='Test Error (SVM)', color='red')
    plt.fill_between(train_sizes, ci_lower, ci_upper, color='red', alpha=0.2, label='95% Confidence Interval')

    # Добавляем горизонтальную линию для ошибки базового классификатора
    plt.axhline(y=baseline_error, color='blue', linestyle='--', label='Baseline Error (Linear Regression)')

    plt.xlabel("Training Set Size")
    plt.ylabel("Test Error")
    plt.title("Test Error Curve with Baseline")
    plt.legend()
    plt.show()

def random_search_svm(X_train, y_train, n_iter=50):
    # Диапазоны гиперпараметров для случайного перебора
    C_range = [0.1, 1, 10, 100]
    degree_range = [2, 3, 4, 5]
    kernel_options = ['linear', 'poly', 'rbf']
    gamma_range = [0.01, 0.1, 1, 10]

    best_accuracy = 0
    best_params = None

    for i in range(n_iter):
        # Случайный выбор гиперпараметров
        C = random.choice(C_range)
        degree = random.choice(degree_range)
        kernel = random.choice(kernel_options)
        gamma = random.choice(gamma_range)

        # Инициализируем и обучаем SVM
        classifier = SVM(kernel=kernel, C=C, degree=degree, gamma=gamma)
        classifier.fit(X_train.values, y_train.values)
        y_pred = classifier.predict(X_train.values)
        accuracy = accuracy_score(y_train, y_pred)

        # Обновляем лучшие гиперпараметры, если текущая точность выше
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = (C, degree, kernel, gamma)
        print(f"{i + 1} random_search_svm")

    print(f"Лучшие гиперпараметры SVM: C={best_params[0]}, degree={best_params[1]}, kernel={best_params[2]}, gamma={best_params[3]}")
    print(f"Лучшая точность на обучении: {best_accuracy:.2f}")
    return best_params


def random_search_gradient(X_train, y_train, n_iter=50):
    # Диапазоны гиперпараметров для случайного перебора
    learning_rate_range = [0.0001, 0.001, 0.01, 0.1]
    alpha_range = [0.1, 0.5, 1, 5]
    lambda_range = [0, 0.1, 0.5, 1]

    best_accuracy = 0
    best_params = None

    for i in range(n_iter):
        # Случайный выбор гиперпараметров
        learning_rate = random.choice(learning_rate_range)
        alpha = random.choice(alpha_range)
        lambda_ = random.choice(lambda_range)

        # Инициализируем и обучаем градиентный классификатор
        classifier = GradientLinearClassifier(learning_rate=learning_rate, alpha=alpha, lambda_=lambda_)
        classifier.fit(X_train.values, y_train.values)
        y_pred = classifier.predict(X_train.values)
        accuracy = accuracy_score(y_train, y_pred)

        # Обновляем лучшие гиперпараметры, если текущая точность выше
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = (learning_rate, alpha, lambda_)
        print(f"{i + 1} random_search_gradient")
    print(f"Лучшие гиперпараметры Gradient Classifier: learning_rate={best_params[0]}, alpha={best_params[1]}, lambda_={best_params[2]}")
    print(f"Лучшая точность на обучении: {best_accuracy:.2f}")
    return best_params

# Лучшие гиперпараметры SVM: C=0.1, degree=4, kernel=rbf, gamma=1
best_svm_params = random_search_svm(X_train, y_train, n_iter=100)
# Лучшие гиперпараметры Gradient Classifier: learning_rate=0.1, alpha=0.1, lambda_=0
best_gradient_params = random_search_gradient(X_train, y_train, n_iter=100)


classifierGrad = GradientLinearClassifier()
classifierSVM = SVM()
classifierMatrix = MatrixLinearClassifier()

classifierGrad.fit(X_train.values, y_train.values)
y_pred_grad = classifierGrad.predict(X_test.values)
accuracy_grad = accuracy_score(y_test, y_pred_grad)
print(f"Точность Gradient Classifier на тесте: {accuracy_grad:.2f}")


classifierSVM.fit(X_train.values, y_train.values)
y_pred_svm = classifierSVM.predict(X_test.values)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"Точность SVM на тесте: {accuracy_svm:.2f}")

classifierMatrix.fit(X_train.values, y_train.values)
y_pred_matrix = classifierMatrix.predict(X_test.values)
accuracy_matrix = accuracy_score(y_test, y_pred_matrix)
print(f"Точность Matrix Classifier на тесте: {accuracy_matrix:.2f}")


print("Построение кривой обучения для GradientLinearClassifier...")
plot_learning_curve(classifierGrad, X_train, y_train)

print("Построение кривой обучения для SVM...")
plot_learning_curve(classifierSVM, X_train, y_train)

print("Построение кривой ошибки на тестовом множестве для GradientLinearClassifier...")
plot_test_error_curve(classifierGrad, X, y)

print("Построение кривой ошибки на тестовом множестве для SVM...")
plot_test_error_curve(classifierSVM, X, y)

print("Построение кривой ошибки на тестовом множестве для MatrixLinearClassifier...")
plot_test_error_curve_with_baseline(classifierMatrix, X, y)
