import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from numpy.linalg import norm


class KernelSVM:
    def __init__(self, kernel='rbf', C=1.0, gamma='scale', degree=3, coef0=0.0,
                 tol=1e-3, max_iter=1000, random_state=None):
        """
        Инициализация параметров SVM

        Параметры:
        ----------
        kernel : str
            Тип ядра ('linear', 'rbf', 'poly', 'sigmoid')
        C : float
            Параметр регуляризации
        gamma : float или 'scale'
            Коэффициент ядра для 'rbf', 'poly', 'sigmoid'
        degree : int
            Степень полиномиального ядра
        coef0 : float
            Свободный член для полиномиального и сигмоидного ядра
        tol : float
            Допуск для условий оптимальности
        max_iter : int
            Максимальное количество итераций
        random_state : int или None
            Начальное значение для генератора случайных чисел
        """
        self.kernel_type = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state

        # Определяем доступные функции ядра
        self.kernel_functions = {
            'linear': self._linear_kernel,
            'rbf': self._rbf_kernel,
            'poly': self._polynomial_kernel,
            'sigmoid': self._sigmoid_kernel
        }

    def _linear_kernel(self, x1, x2):
        """Линейное ядро: K(x1, x2) = x1ᵀx2"""
        return np.dot(x1, x2)

    def _rbf_kernel(self, x1, x2):
        """RBF (Гауссово) ядро: K(x1, x2) = exp(-γ||x1-x2||²)"""
        return np.exp(-self.gamma * norm(x1 - x2) ** 2)

    def _polynomial_kernel(self, x1, x2):
        """Полиномиальное ядро: K(x1, x2) = (γx1ᵀx2 + coef0)^degree"""
        return (self.gamma * np.dot(x1, x2) + self.coef0) ** self.degree

    def _sigmoid_kernel(self, x1, x2):
        """Сигмоидное ядро: K(x1, x2) = tanh(γx1ᵀx2 + coef0)"""
        return np.tanh(self.gamma * np.dot(x1, x2) + self.coef0)

    def _compute_kernel_matrix(self, X1, X2=None):
        """Вычисление матрицы ядра K(X1, X2)"""
        if X2 is None:
            X2 = X1

        kernel_function = self.kernel_functions[self.kernel_type]
        n1, n2 = X1.shape[0], X2.shape[0]
        K = np.zeros((n1, n2))

        for i in range(n1):
            for j in range(n2):
                K[i, j] = kernel_function(X1[i], X2[j])
        return K

    def fit(self, X, y):
        """
        Обучение модели SVM методом SMO

        Параметры:
        ----------
        X : array-like
            Обучающие данные
        y : array-like
            Метки классов
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.X = np.array(X)
        self.y = np.array(y)

        if len(np.unique(y)) != 2:
            raise ValueError("Модель поддерживает только бинарную классификацию")

        self.y = np.where(self.y == np.unique(y)[0], -1, 1)

        n_samples = self.X.shape[0]

        if self.gamma == 'scale':
            self.gamma = 1.0 / (self.X.shape[1] * self.X.var())

        self.alphas = np.zeros(n_samples)
        self.bias = 0.0

        self.K = self._compute_kernel_matrix(self.X)

        n_changed = 0
        examine_all = True

        while n_changed > 0 or examine_all:
            n_changed = 0

            if examine_all:
                for i in range(n_samples):
                    n_changed += self._examine_example(i)
            else:
                alphas_nonzero = np.where((self.alphas > 0) & (self.alphas < self.C))[0]
                for i in alphas_nonzero:
                    n_changed += self._examine_example(i)

            if examine_all:
                examine_all = False
            elif n_changed == 0:
                examine_all = True

        sv_mask = self.alphas > 1e-5
        self.support_vectors_ = self.X[sv_mask]
        self.support_vector_labels_ = self.y[sv_mask]
        self.support_vector_alphas_ = self.alphas[sv_mask]

    def _examine_example(self, i2):
        """Проверка примера на нарушение условий оптимальности"""
        y2 = self.y[i2]
        alpha2 = self.alphas[i2]
        E2 = self._error(i2)
        r2 = E2 * y2

        if (r2 < -self.tol and alpha2 < self.C) or (r2 > self.tol and alpha2 > 0):
            if len(self.alphas[(self.alphas > 0) & (self.alphas < self.C)]) > 1:
                i1 = self._select_second(i2, E2)
                if self._take_step(i1, i2):
                    return 1

            for i1 in np.roll(np.where((self.alphas > 0) & (self.alphas < self.C))[0],
                              np.random.randint(self.X.shape[0])):
                if self._take_step(i1, i2):
                    return 1

            for i1 in np.roll(range(self.X.shape[0]), np.random.randint(self.X.shape[0])):
                if self._take_step(i1, i2):
                    return 1

        return 0

    def _take_step(self, i1, i2):
        """Обновляет множители Лагранжа alphas[i1] и alphas[i2]"""
        if i1 == i2:
            return False

        alpha1, alpha2 = self.alphas[i1], self.alphas[i2]
        y1, y2 = self.y[i1], self.y[i2]
        E1, E2 = self._error(i1), self._error(i2)
        s = y1 * y2

        if y1 != y2:
            L = max(0, alpha2 - alpha1)
            H = min(self.C, self.C + alpha2 - alpha1)
        else:
            L = max(0, alpha1 + alpha2 - self.C)
            H = min(self.C, alpha1 + alpha2)

        if L == H:
            return False

        k11, k12, k22 = self.K[i1, i1], self.K[i1, i2], self.K[i2, i2]
        eta = k11 + k22 - 2 * k12

        if eta > 0:
            a2 = alpha2 + y2 * (E1 - E2) / eta
            a2 = np.clip(a2, L, H)
        else:
            return False

        if abs(a2 - alpha2) < 1e-4:
            return False

        a1 = alpha1 + s * (alpha2 - a2)

        self.alphas[i1], self.alphas[i2] = a1, a2

        return True

    def _error(self, i):
        return self._decision_function(self.X[i]) - self.y[i]

    def _decision_function(self, X):
        K = self._compute_kernel_matrix(X, self.X)
        return np.sum(self.alphas * self.y * K.T, axis=1) + self.bias

    def predict(self, X):
        return np.sign(self._decision_function(np.array(X)))


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

classifierSVM= KernelSVM()
classifieSVM.fit(X_train.values, y_train.values)
y_pred3 = classifierSVM.predict(X_test.values)
accuracy3 = accuracy_score(y_test, y_pred3)
print(f"Accuracy matrix: {accuracy3:.2f}")