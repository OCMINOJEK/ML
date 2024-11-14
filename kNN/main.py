from operator import truediv

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import math
import statsmodels.api as sm
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin
import random
from scipy.stats import randint
import matplotlib.pyplot as plt


class KNearestNeighbors:
    def __init__(self, k=None, metric='cosine', h=100, metric_p=1, kernel='uniform', window_type='variable', a=1, b=1):
        """
        :param k: {p_int}
        :param metric: {minkowski, cosine}
        :param h: {p_int}
        :param metric_p: {1, 2, p_int}
        :param kernel: {uniform, gaussian, triangular, epanechnikov, general}
        :param window_type: {fixed, variable}
        :param a: {p_int}
        :param b: {p_int}
        """

        self.y_train_data = None
        self.neigh = None
        self.y_train = None
        self.X_train = None
        self.X_train_data = None

        self.k = k
        self.h = h
        self.metric = metric
        self.metric_p = metric_p
        self.window_type = window_type
        self.kernel = kernel
        self.a = a
        self.b = b
        self.is_lowess = True

    def fit(self, X, y):
        self.X_train_data = X
        self.y_train_data = y
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        self.neigh = NearestNeighbors(n_neighbors = len(X), metric=self._compute_distance)
        self.neigh.fit(self.X_train)

    def lowess(self):
        for index, row in self.X_train_data.iterrows():
            x_i = row.to_numpy()
            X = np.array(self.X_train_data.drop(index))
            neigh = NearestNeighbors(n_neighbors=len(X), metric=self._compute_distance)
            neigh.fit(X)
            if self.window_type == 'variable':
                distances, indices = neigh.kneighbors([x_i], n_neighbors=self.k + 1, return_distance=True)
            elif self.window_type == 'fixed':
                distances, indices = neigh.radius_neighbors([x_i], radius=self.h, return_distance=True)
            else:
                raise Exception('Invalid window type')
            distances = distances[0]
            indices = indices[0]
            class_count = {}
            for i in range(len(distances)):
                label = self.y_train[indices[i]]
                if label in class_count:
                    class_count[label] += 1
                else:
                    class_count[label] = 1
            self.X_train_data.loc[index, "weights"] = self._kernel(1 - (class_count[self.y_train_data.loc[index]] / len(distances)))
        self.X_train = np.array(self.X_train_data)
        self.neigh.fit(self.X_train)

    def predict(self, X):
        X = np.array(X)
        predicted_classes = []
        for i in range(len(X)):
            predicted_classes.append(self._predict_single(X[i]))
        return predicted_classes

    def _predict_single(self, x):
        weighted_votes = self.calculate_weights(x)

        predicted_class = max(weighted_votes, key=weighted_votes.get)
        return predicted_class

    def calculate_weights(self, x):
        distances, indices = self.neigh.kneighbors([x], return_distance=True)
        distances = distances[0]
        indices = indices[0]
        weighted_votes = {}

        if self.window_type == 'variable':
            h = distances[self.k]
        elif self.window_type == 'fixed':
            h = self.h
        else:
            raise Exception("window_type error")

        for j in range(len(distances)):
            if h == 0:
                h = 1e-5
            weight = self._kernel(distances[j] / h) * self.X_train[indices[j], -1]
            label = self.y_train[indices[j]]
            if label in weighted_votes:
                weighted_votes[label] += weight
            else:
                weighted_votes[label] = weight
        return weighted_votes

    def _compute_distance(self, x_test, x_train):
        if self.metric == 'minkowski':
            return np.power(np.sum(np.abs(x_test - x_train) ** self.metric_p), 1 / self.metric_p)
        elif self.metric == 'cosine':
            numerator = np.dot(x_test, x_train)
            denominator_1 = np.linalg.norm(x_train)
            denominator_2 = np.linalg.norm(x_test)
            if denominator_1 == 0 or denominator_2 == 0:
                return 1.0
            return 1 - (numerator / (denominator_1 * denominator_2))
        else:
            raise ValueError("Unknown metric type")

    def _kernel(self, r):
        if self.kernel == 'uniform':
            return 0.5 if abs(r) < 1 else 0.0
        elif self.kernel == 'gaussian':
            return (1 / (2 * math.sqrt(math.pi))) * np.exp(-0.5 * r ** 2)
        elif self.kernel == 'triangular':
            return 1 - abs(r) if abs(r) < 1 else 0
        elif self.kernel == 'epanechnikov':
            return (3/4) * (1 - r ** 2) if abs(r) < 1 else 0
        elif self.kernel == 'general':
            return (1 - abs(r) ** self.a) ** self.b if abs(r) < 1 else 0
        else:
            raise ValueError("Unknown kernel type")





data = pd.read_csv("processed_data.csv")

# Добавляем столбец с априорными весами
data['weights'] = 1.0  # Устанавливаем начальные веса для всех объектов равными 1.0
# data.loc[100:300, 'weights'] = 0.5 #пример
y = data['Rank']
X = data.drop(columns=['Rank'])
X = X.drop(X.columns[43:-1], axis=1)  # Удаляем favorite champions
X = X.drop(X.columns[37:-1], axis=1)  # Удаляем favorite roles

X = X.head(1000)
y = y.head(1000)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

def random_param():
    param_distributions = {
        'k': randint(5, 500),
        'metric': ['minkowski', 'cosine'],
        'metric_p': randint(1, 100),
        'kernel': ['uniform', 'gaussian', 'triangular', 'epanechnikov', 'general'],
        'window_type': ['fixed', 'variable'],
        'h': randint(1, 2000),
        'a': randint(1, 20),
        'b': randint(1, 20)
    }
    random_param_sets = []
    for i in range(100):
        param_set = {
            'k': param_distributions['k'].rvs(),
            'metric': random.choice(param_distributions['metric']),
            'metric_p': param_distributions['metric_p'].rvs(),
            'kernel': random.choice(param_distributions['kernel']),
            'window_type': random.choice(param_distributions['window_type']),
            'h': param_distributions['h'].rvs(),
            'a': param_distributions['a'].rvs(),
            'b': param_distributions['b'].rvs()
        }
        knn = KNearestNeighbors(**param_set)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        random_param_sets.append((param_set, accuracy))
        print(f"Accuracy {i + 1} :", accuracy)
    sorted_param_sets = sorted(random_param_sets, key=lambda x: x[1], reverse=True)
    best_params, best_accuracy = sorted_param_sets[0]
    print("Best Accuracy:", best_accuracy)
    print("Best Hyperparameters:", best_params)



knn = KNearestNeighbors(k = 41, metric_p=37, metric='minkowski', kernel='epanechnikov', window_type='variable')
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
knn.lowess()
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)




# fixed_params_for_k = {
#     'metric': 'minkowski',
#     'metric_p': 37,
#     'kernel': 'triangular',
#     'window_type': 'variable',
#     'h': 100,
#     'a': 1,
#     'b': 1
# }
#
# train_accuracies_k, test_accuracies_k = [], []
# train_accuracies_h, test_accuracies_h = [], []
#
# k_values = range(10, 220, 10)
#
# for k in k_values:
#     knn = KNearestNeighbors(k=k, **fixed_params_for_k)
#     knn.fit(X_train, y_train)
#
#     y_train_pred = knn.predict(X_train)
#     train_accuracy_k = accuracy_score(y_train, y_train_pred)
#     train_accuracies_k.append(train_accuracy_k)
#
#     y_test_pred = knn.predict(X_test)
#     test_accuracy_k = accuracy_score(y_test, y_test_pred)
#     test_accuracies_k.append(test_accuracy_k)
#     print(train_accuracy_k, test_accuracy_k)
# print("aboba")
#
# # Построение графика
# plt.figure(figsize=(12, 8))
# # Линии для k
# plt.plot(k_values, train_accuracies_k, label='Train Accuracy (varying k)', color='blue', linestyle='--', marker='o')
# plt.plot(k_values, test_accuracies_k, label='Test Accuracy (varying k)', color='blue', linestyle='-', marker='o')
#
# # Оформление графика
# plt.xlabel('Parameter Value k')
# plt.ylabel('Accuracy')
# plt.title('Accuracy k')
# plt.legend()
# plt.grid(True)
# plt.show()


