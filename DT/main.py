import numpy as np
import pandas as pd
from pyexpat import features
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt


class RandomForest:
    def __init__(self, criterion='gini', repeats = 9):
        self.criterion = criterion
        self.repeats = repeats
        self.list_features_indices = []
        self.trees = []

    def fit(self, X, y):
        n, p = X.shape
        for i in range(self.repeats):
            X_sample, y_sample = resample(X, y, replace=True)
            features_indices = np.random.choice(p, int(np.sqrt(p)), replace=False)
            self.list_features_indices.append(features_indices)

            tree = DecisionTree(max_depth=None, min_samples_split = 0, max_leaf_nodes = None, criterion=self.criterion)
            tree.fit(X_sample[X.columns[features_indices]], y_sample)
            self.trees.append(tree)
    def predict(self, X):
        predictions = []
        for tree, feature in zip(self.trees, self.list_features_indices):
            predictions.append(tree.predict(X[X.columns[feature]]))
        matrix = np.array([predictions[i] for i in range(self.repeats)]).T
        ans_predictions = []
        for i in range(len(matrix)):
            ans_predictions.append(np.unique(matrix[i], return_counts=True)[0][np.unique(matrix[i], return_counts=True)[1].argmax()])
        return ans_predictions


def _split_data(X, y, f, t):
    f = X.columns.get_loc(f)
    X_np = X.values
    y_np = y.values

    left_indices = X_np[:, f] <= t
    right_indices = X_np[:, f] > t

    X_left_np, X_right_np = X_np[left_indices], X_np[right_indices]
    y_left_np, y_right_np = y_np[left_indices], y_np[right_indices]

    X_left = pd.DataFrame(X_left_np, columns=X.columns)
    X_right = pd.DataFrame(X_right_np, columns=X.columns)
    y_left = pd.Series(y_left_np, index=X.index[left_indices])
    y_right = pd.Series(y_right_np, index=X.index[right_indices])

    return X_left, X_right, y_left, y_right


def _entropy(y):
    classes, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return -np.sum(p * np.log2(p))


def _gini(y):
    classes, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return 1 - np.sum(p ** 2)


class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=5, max_leaf_nodes=None, criterion='gini'):
        self.n_leaves = None
        self.tree = None
        self.min_samples_split = min_samples_split  # минимальное количество объектов в узле
        self.max_depth = max_depth  # максимальная глубина
        self.criterion = criterion  # выбор критерия
        self.max_leaf_nodes = max_leaf_nodes  # максимальное количество листьев
        self.height = 0

    def _split(self, X, y):
        feature, threshold, criterion = None, None, float('inf')
        for f in X.columns:  # f - текущий признак
            t_s = np.unique(X[f])
            for t in t_s:  # t - порог для f
                X_left, X_right, y_left, y_right = _split_data(X, y, f, t)
                if len(y_left) < self.min_samples_split or len(y_right) < self.min_samples_split:
                    continue
                if self.criterion == 'gini':
                    impurity_left = _gini(y_left)
                    impurity_right = _gini(y_right)
                elif self.criterion == 'entropy':
                    impurity_left = _entropy(y_left)
                    impurity_right = _entropy(y_right)
                else:
                    raise Exception('Invalid criterion')
                new_criterion = (len(y_left) / len(y) * impurity_left + len(y_right) / len(y) * impurity_right)
                if new_criterion < criterion:
                    criterion = new_criterion
                    feature = f
                    threshold = t
        return feature, threshold

    def _build_tree(self, X, y, depth=0):
        if (len(np.unique(y)) == 1 or
            self.max_depth is not None and depth >= self.max_depth) or \
                len(y) < self.min_samples_split or \
                self.max_leaf_nodes is not None and self.n_leaves >= self.max_leaf_nodes:
            self.n_leaves += 1
            self.height = max(self.height, depth + 1)
            return np.unique(y, return_counts=True)[0][np.unique(y, return_counts=True)[1].argmax()]
        feature, threshold = self._split(X, y)
        if feature is None:
            self.n_leaves += 1
            self.height = max(self.height, depth + 1)
            return np.unique(y, return_counts=True)[0][np.unique(y, return_counts=True)[1].argmax()]
        X_left, X_right, y_left, y_right = _split_data(X, y, feature, threshold)
        if len(X_left) == len(X) or len(X_right) == len(X):
            self.height = max(self.height, depth + 1)
            return np.unique(y, return_counts=True)[0][np.unique(y, return_counts=True)[1].argmax()]
        return {
            'depth': depth,
            'feature': feature,
            'threshold': threshold,
            'left': self._build_tree(X_left, y_left, depth + 1),
            'right': self._build_tree(X_right, y_right, depth + 1),
        }

    def fit(self, X, y):
        self.n_leaves = 0
        self.tree = self._build_tree(X, y)

    def _predict(self, x, tree):
        if not isinstance(tree, dict):
            return tree
        if x[tree['feature']] <= tree['threshold']:
            return self._predict(x, tree['left'])
        else:
            return self._predict(x, tree['right'])

    def predict(self, X):
        return np.array([self._predict(X.iloc[i], self.tree) for i in range(len(X))])


data = pd.read_csv("processed_data.csv")
y = data['Rank']
X = data.drop(columns=['Rank'])
X = X.drop(X.columns[42:], axis=1)  # Удаляем favorite champions
X = X.drop(X.columns[37:], axis=1)  # Удаляем favorite roles

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.97, random_state=42)

# dt = DecisionTree()
# dt.fit(X_train, y_train)
# y_pred = dt.predict(X_test)
# print(accuracy_score(y_test, y_pred))

# rf = RandomForest()
# rf.fit(X_train, y_train)
# y_pred = rf.predict(X_test)
# print(accuracy_score(y_test, y_pred))


def library_tree_height_vs_params(X_train, y_train):
    a = []
    b = []
    c = []
    min_samples_splits = [2, 5, 10, 20, 50, 100]
    max_depths = [5, 10, 20, 50, 100]
    max_leaf_nodes = [5, 10, 20, 50, 100]

    for min_samples_split in min_samples_splits:
        clf = DecisionTreeClassifier(min_samples_split=min_samples_split, random_state=42)
        clf.fit(X_train, y_train)
        a.append(clf.tree_.max_depth)

    plt.figure(figsize=(8, 6))
    plt.plot(min_samples_splits, a, marker='o', label='Library Decision Tree')
    plt.title("Tree Height vs min_samples_split (Library Implementation)")
    plt.xlabel("min_samples_split")
    plt.ylabel("Tree Height")
    plt.grid()
    plt.legend()
    plt.show()

    for max_depth in max_depths:
        clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        clf.fit(X_train, y_train)
        b.append(clf.tree_.max_depth)

    plt.figure(figsize=(8, 6))
    plt.plot(max_depths, b, marker='o', label='Library Decision Tree')
    plt.title("Tree Height vs max_depth (Library Implementation)")
    plt.xlabel("max_depth")
    plt.ylabel("Tree Height")
    plt.grid()
    plt.legend()
    plt.show()

    for max_leaf_node in max_leaf_nodes:
        clf = DecisionTreeClassifier(max_leaf_nodes=max_leaf_node, random_state=42)
        clf.fit(X_train, y_train)
        c.append(clf.tree_.max_depth)

    plt.figure(figsize=(8, 6))
    plt.plot(max_leaf_nodes, c, marker='o', label='Library Decision Tree')
    plt.title("Tree Height vs max_leaf_nodes (Library Implementation)")
    plt.xlabel("max_leaf_nodes")
    plt.ylabel("Tree Height")
    plt.grid()
    plt.legend()
    plt.show()



def custom_tree_height_vs_params(X_train, y_train):
    a = []
    b = []
    c = []
    min_samples_splits = [2, 5, 10, 20, 50, 100]
    max_depths = [5, 10, 20, 50, 100]
    max_leaf_nodes = [5, 10, 20, 50, 100]

    for min_samples_split in min_samples_splits:
        clf = DecisionTree(min_samples_split=min_samples_split)
        clf.fit(X_train, y_train)
        a.append(clf.height)

    plt.figure(figsize=(8, 6))
    plt.plot(min_samples_splits, a, marker='o', label='Library Decision Tree')
    plt.title("Tree Height vs min_samples_split (Custom Implementation)")
    plt.xlabel("min_samples_split")
    plt.ylabel("Tree Height")
    plt.grid()
    plt.legend()
    plt.show()

    for max_depth in max_depths:
        clf = DecisionTree(max_depth=max_depth)
        clf.fit(X_train, y_train)
        b.append(clf.height)

    plt.figure(figsize=(8, 6))
    plt.plot(max_depths, b, marker='o', label='Library Decision Tree')
    plt.title("Tree Height vs max_depth (Custom Implementation)")
    plt.xlabel("max_depth")
    plt.ylabel("Tree Height")
    plt.grid()
    plt.legend()
    plt.show()

    for max_leaf_node in max_leaf_nodes:
        clf = DecisionTree(max_leaf_nodes=max_leaf_node)
        clf.fit(X_train, y_train)
        c.append(clf.height)

    plt.figure(figsize=(8, 6))
    plt.plot(max_leaf_nodes, c, marker='o', label='Library Decision Tree')
    plt.title("Tree Height vs max_leaf_nodes (Custom Implementation)")
    plt.xlabel("max_leaf_nodes")
    plt.ylabel("Tree Height")
    plt.grid()
    plt.legend()
    plt.show()

def error_vs_height(X_train, X_test, y_train, y_test):
    heights_custom = []
    train_acc_custom = []
    test_acc_custom = []

    heights_lib = []
    train_acc_lib = []
    test_acc_lib = []

    min_samples_splits = [2, 5, 10, 20, 50, 100]

    for min_samples_split in min_samples_splits:
        print(min_samples_split)
        dt = DecisionTree(min_samples_split=min_samples_split)
        dt.fit(X_train, y_train)
        y_train_pred = dt.predict(X_train)
        y_test_pred = dt.predict(X_test)
        heights_custom.append(dt.height)
        train_acc_custom.append(accuracy_score(y_train, y_train_pred))
        test_acc_custom.append(accuracy_score(y_test, y_test_pred))


    for min_samples_split in min_samples_splits:
        ldt = DecisionTreeClassifier(min_samples_split=min_samples_split, random_state=42)
        ldt.fit(X_train, y_train)
        y_train_pred = ldt.predict(X_train)
        y_test_pred = ldt.predict(X_test)
        heights_lib.append(ldt.tree_.max_depth)
        train_acc_lib.append(accuracy_score(y_train, y_train_pred))
        test_acc_lib.append(accuracy_score(y_test, y_test_pred))

    plt.figure(figsize=(12, 8))
    plt.plot(heights_custom, train_acc_custom, label='Custom Tree - Train Accuracy', marker='o')
    plt.plot(heights_custom, test_acc_custom, label='Custom Tree - Test Accuracy', marker='o')
    plt.plot(heights_lib, train_acc_lib, label='Library Tree - Train Accuracy', marker='x')
    plt.plot(heights_lib, test_acc_lib, label='Library Tree - Test Accuracy', marker='x')
    plt.title("Accuracy vs Tree Height")
    plt.xlabel("Tree Height")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.show()

def error_vs_num_trees(X_train, X_test, y_train, y_test):
    train_acc_custom = []
    test_acc_custom = []

    train_acc_lib = []
    test_acc_lib = []

    num_trees = [i for i in range(1, 21)]

    for n in num_trees:
        print(n)
        rf = RandomForest(repeats=n)
        rf.fit(X_train, y_train)
        y_train_pred = rf.predict(X_train)
        y_test_pred = rf.predict(X_test)
        train_acc_custom.append(accuracy_score(y_train, y_train_pred))
        test_acc_custom.append(accuracy_score(y_test, y_test_pred))

    for n in num_trees:
        print(n)
        clf = RandomForestClassifier(n_estimators=n, random_state=42)
        clf.fit(X_train, y_train)
        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)
        train_acc_lib.append(accuracy_score(y_train, y_train_pred))
        test_acc_lib.append(accuracy_score(y_test, y_test_pred))

    plt.figure(figsize=(12, 8))
    plt.plot(num_trees, train_acc_custom, label='Custom Forest - Train Accuracy', marker='o')
    plt.plot(num_trees, test_acc_custom, label='Custom Forest - Test Accuracy', marker='o')
    plt.plot(num_trees, train_acc_lib, label='Library Forest - Train Accuracy', marker='x')
    plt.plot(num_trees, test_acc_lib, label='Library Forest - Test Accuracy', marker='x')
    plt.title("Accuracy vs Number of Trees")
    plt.xlabel("Number of Trees")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.show()

def error_vs_num_trees_boosting(X_train, X_test, y_train, y_test):
    train_acc_custom = []
    test_acc_custom = []

    train_acc_lib = []
    test_acc_lib = []

    num_trees = [i for i in range(1, 21)]

    for n in num_trees:
        print(n)
        rf = RandomForest(repeats=n)
        rf.fit(X_train, y_train)
        y_train_pred = rf.predict(X_train)
        y_test_pred = rf.predict(X_test)
        train_acc_custom.append(accuracy_score(y_train, y_train_pred))
        test_acc_custom.append(accuracy_score(y_test, y_test_pred))

    for n in num_trees:
        gb = GradientBoostingClassifier(n_estimators=n, random_state=42)
        gb.fit(X_train, y_train)
        y_train_pred = gb.predict(X_train)
        y_test_pred = gb.predict(X_test)
        train_acc_lib.append(accuracy_score(y_train, y_train_pred))
        test_acc_lib.append(accuracy_score(y_test, y_test_pred))

    plt.figure(figsize=(12, 8))
    plt.plot(num_trees, train_acc_custom, label='Custom Forest - Train Accuracy', marker='o')
    plt.plot(num_trees, test_acc_custom, label='Custom Forest - Test Accuracy', marker='o')
    plt.plot(num_trees, train_acc_lib, label='Library Forest - Train Accuracy', marker='x')
    plt.plot(num_trees, test_acc_lib, label='Library Forest - Test Accuracy', marker='x')
    plt.title("Accuracy vs Number of Trees (Gradient Boosting)")
    plt.xlabel("Number of Trees")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.show()

library_tree_height_vs_params(X_train, y_train)
custom_tree_height_vs_params(X_train, y_train)
error_vs_height(X_train, X_test, y_train, y_test)
error_vs_num_trees(X_train, X_test, y_train, y_test)
error_vs_num_trees_boosting(X_train, X_test, y_train, y_test)