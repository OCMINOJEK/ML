import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2, RFE, mutual_info_classif
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance




class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=5, max_leaf_nodes=None, criterion='gini'):
        self.n_leaves = None
        self.tree = None
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.criterion = criterion
        self.max_leaf_nodes = max_leaf_nodes
        self.height = 0

    @staticmethod
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

    @staticmethod
    def _entropy(y):
        classes, counts = np.unique(y, return_counts=True)
        p = counts / len(y)
        return -np.sum(p * np.log2(p))

    @staticmethod
    def _gini(y):
        classes, counts = np.unique(y, return_counts=True)
        p = counts / len(y)
        return 1 - np.sum(p ** 2)

    def compute_feature_important(self, n_features):
        important = np.zeros(n_features)
        self._compute_feature_important_recursive(self.tree, important)
        return important / important.sum()

    def _compute_feature_important_recursive(self, node, important):
        if isinstance(node, dict):
            feature_idx = self.X.columns.get_loc(node['feature'])
            important[feature_idx] += 1 / (node['depth'] + 1)
            self._compute_feature_important_recursive(node['left'], important)
            self._compute_feature_important_recursive(node['right'], important)

    def _split(self, X, y):
        feature, threshold, criterion = None, None, float('inf')
        for f in X.columns:
            t_s = np.unique(X[f])
            for t in t_s:
                X_left, X_right, y_left, y_right = self._split_data(X, y, f, t)
                if len(y_left) < self.min_samples_split or len(y_right) < self.min_samples_split:
                    continue
                if self.criterion == 'gini':
                    impurity_left = self._gini(y_left)
                    impurity_right = self._gini(y_right)
                elif self.criterion == 'entropy':
                    impurity_left = self._entropy(y_left)
                    impurity_right = self._entropy(y_right)
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
        X_left, X_right, y_left, y_right = self._split_data(X, y, feature, threshold)
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
        self.X = X
        self.y = y
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

class RandomForest:
    def __init__(self, criterion='gini', repeats=9):
        self.criterion = criterion
        self.repeats = repeats
        self.list_features_indices = []
        self.trees = []
        self.feature_important_ = None

    def fit(self, X, y):
        n, p = X.shape
        feature_important = np.zeros(p)
        for i in range(self.repeats):
            X_sample, y_sample = resample(X, y, replace=True)
            features_indices = np.random.choice(p, int(np.sqrt(p)), replace=False)
            self.list_features_indices.append(features_indices)

            tree = DecisionTree(max_depth=None, min_samples_split=0, max_leaf_nodes=None, criterion=self.criterion)
            tree.fit(X_sample[X.columns[features_indices]], y_sample)
            self.trees.append(tree)

            tree_important = tree.compute_feature_important(len(features_indices))
            for idx, important in zip(features_indices, tree_important):
                feature_important[idx] += important
        self.feature_important_ = feature_important / self.repeats

    def predict(self, X):
        predictions = []
        for tree, feature in zip(self.trees, self.list_features_indices):
            predictions.append(tree.predict(X[X.columns[feature]]))
        matrix = np.array([predictions[i] for i in range(self.repeats)]).T
        ans_predictions = []
        for i in range(len(matrix)):
            ans_predictions.append(np.unique(matrix[i], return_counts=True)[0][np.unique(matrix[i], return_counts=True)[1].argmax()])
        return ans_predictions

class SFS:
    def __init__(self, model, k_features):
        self.model = model
        self.k_features = k_features
        self.selected_features = []

    def fit(self, X, y):
        features = list(X.columns)
        count = 1
        while len(self.selected_features) < self.k_features:
            best_feature = None
            best_score = 0
            count_f = 0
            for feature in features:
                current_features = self.selected_features + [feature]
                X_subset = X[current_features]
                self.model.fit(X_subset, y)
                y_pred = self.model.predict(X_subset)
                score = accuracy_score(y, y_pred)
                if score > best_score:
                    best_score = score
                    best_feature = feature
                count_f += 1
            print(count)
            count += 1
            self.selected_features.append(best_feature)
            features.remove(best_feature)
        return self

    def transform(self, X):
        return X[self.selected_features]

class SpearmanFilter:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        correlations = []
        for feature in X.columns:
            if len(X[X[feature] == 1]) == 0:
                continue
            corr, _ = spearmanr(X[feature], y)
            correlations.append((feature, abs(corr)))
        self.top_features = sorted(correlations, key=lambda x: x[1], reverse=True)[:self.k]
        self.selected_features = [f[0] for f in self.top_features]
        return self

    def transform(self, X):
        return X[self.selected_features]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)




data = pd.read_csv("SMS.tsv", sep='\t')
X = data["text"]
y = data["class"].map({"ham": 0, "spam": 1})
k_features = 30

# Преобразование текста в CV
vectorizer = CountVectorizer()
X_CV = vectorizer.fit_transform(X)
feature_names = vectorizer.get_feature_names_out()
df_CV = pd.DataFrame(X_CV.toarray(), columns=feature_names)

X_train, X_test, y_train, y_test = train_test_split(df_CV, y, test_size=0.80, random_state=43)


# Встроенный метод
rf_model = RandomForest(criterion='gini', repeats=10)
rf_model.fit(X_train, y_train)
important_features_indices = np.argsort(rf_model.feature_important_)[-k_features:]
# X_train_embedded = X_train.iloc[:, important_features_indices]
# X_test_embedded = X_test.iloc[:, important_features_indices]

# embedded_model = DecisionTree()
# embedded_model.fit(X_train_embedded, y_train)
# y_pred_embedded = embedded_model.predict(X_test_embedded)
# accuracy_embedded = accuracy_score(y_test, y_pred_embedded)
#
# print("Accuracy (встроенный метод):", accuracy_embedded)


# Оберточный метод
sfs = SFS(DecisionTreeClassifier(), k_features)
sfs.fit(X_train, y_train)
# X_train_wrapper = sfs.transform(X_train)
# X_test_wrapper = sfs.transform(X_test)

# wrapper_model = DecisionTree()
# wrapper_model.fit(X_train_sfs, y_train)
# y_pred_wrapper = wrapper_model.predict(X_test_sfs)
# accuracy_wrapper = accuracy_score(y_test, y_pred_wrapper)
# #Accuracy (обёрточный метод): 0.8934499775684164
# print("Accuracy (обёрточный метод):", accuracy_wrapper)

# Фильтрующий метод
spearman_filter = SpearmanFilter(k=k_features)
X_train_filter = spearman_filter.fit_transform(X_train, y_train)
# X_test_filter = spearman_filter.transform(X_test)
#
# spearman_model = DecisionTree()
# spearman_model.fit(X_train_spearman, y_train)
# y_pred_spearman = spearman_model.predict(X_test_spearman)
# accuracy_spearman = accuracy_score(y_test, y_pred_spearman)
#
# print("Accuracy (фильтрующий метод):", accuracy_spearman)

# Встроенный метод (Embedded Method) с L1-регуляризацией
l1_model = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
l1_model.fit(X_train, y_train)
embedded_features_indices = np.where(l1_model.coef_[0] != 0)[0][:k_features]
X_train_embedded_lib = X_train.iloc[:, embedded_features_indices]
X_test_embedded_lib = X_test.iloc[:, embedded_features_indices]


# embedded_model = LogisticRegression()
# embedded_model.fit(X_train_embedded, y_train)
# y_pred_embedded = embedded_model.predict(X_test_embedded)
# accuracy_embedded = accuracy_score(y_test, y_pred_embedded)
# print("Accuracy (встроенный метод - библиотека, L1-регуляризация):", accuracy_embedded)

# Обёрточный метод библиотечный (перестановочная полезность)
base_model = DecisionTreeClassifier()
base_model.fit(X_train, y_train)
perm_importance = permutation_importance(base_model, X_train, y_train, scoring="accuracy", random_state=42)
perm_sorted_idx = perm_importance.importances_mean.argsort()[-k_features:]
X_train_wrapper_lib = X_train.iloc[:, perm_sorted_idx]
X_test_wrapper_lib = X_test.iloc[:, perm_sorted_idx]
#
# wrapper_model = DecisionTreeClassifier(random_state=42)
# wrapper_model.fit(X_train_permutation, y_train)
# y_pred_wrapper = wrapper_model.predict(X_test_permutation)
# accuracy_wrapper = accuracy_score(y_test, y_pred_wrapper)
# print("Accuracy (обёрточный метод - перестановочная полезность):", accuracy_wrapper)

# Фильтрующий библиотечный метод (mutual_info_classif)
select_k_best = SelectKBest(score_func=mutual_info_classif, k=k_features)
X_train_filter_lib = select_k_best.fit_transform(X_train, y_train)
X_test_filter_lib = select_k_best.transform(X_test)

# filter_model = DecisionTreeClassifier(random_state=42)
# filter_model.fit(X_train_filter, y_train)
# y_pred_filter = filter_model.predict(X_test_filter)
# accuracy_filter = accuracy_score(y_test, y_pred_filter)
# print("Accuracy (фильтрующий метод - mutual_info_classif):", accuracy_filter)

embedded_features = X_train.columns[important_features_indices]
wrapper_features = X_train.columns[sfs.selected_features]
filter_features = spearman_filter.selected_features
l1_features = X_train.columns[embedded_features_indices]
wrapper_lib_features = X_train.columns[perm_sorted_idx]
filter_lib_features = X_train.columns[select_k_best.get_support(indices=True)]