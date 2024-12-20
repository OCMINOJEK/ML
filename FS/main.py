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
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


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
            ans_predictions.append(
                np.unique(matrix[i], return_counts=True)[0][np.unique(matrix[i], return_counts=True)[1].argmax()])
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


def methods(X_train, y_train, k_features):
    # Встроенный метод
    rf_model = RandomForest(criterion='gini', repeats=10)
    rf_model.fit(X_train, y_train)
    important_features_indices = np.argsort(rf_model.feature_important_)[-k_features:]
    print("Встроенный метод")

    # Оберточный метод
    sfs = SFS(DecisionTreeClassifier(), k_features)
    sfs.fit(X_train, y_train)
    print("Оберточный метод")

    # Фильтрующий метод
    spearman_filter = SpearmanFilter(k=k_features)
    X_train_filter = spearman_filter.fit_transform(X_train, y_train)
    print("Фильтрующий метод")

    # Встроенный метод (Embedded Method) с L1-регуляризацией
    l1_model = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
    l1_model.fit(X_train, y_train)
    embedded_features_indices = np.where(l1_model.coef_[0] != 0)[0][:k_features]
    print("Встроенный метод (Embedded Method) с L1-регуляризацией")

    # Обёрточный метод библиотечный (RFE)
    rfa = RFE(estimator=LogisticRegression(), n_features_to_select=k_features, step=1)
    rfa.fit(X_train, y_train)
    rfa_idx = X_train.columns[rfa.get_support(indices=True)]
    print("Обёрточный метод библиотечный (RFE)")

    # Фильтрующий библиотечный метод (mutual_info_classif)
    select_k_best = SelectKBest(score_func=mutual_info_classif, k=k_features)
    select_k_best.fit(X_train, y_train)
    print("Фильтрующий библиотечный метод (mutual_info_classif)")

    embedded_features = X_train.columns[important_features_indices]
    wrapper_features = sfs.selected_features
    filter_features = spearman_filter.selected_features
    l1_features = X_train.columns[embedded_features_indices]
    wrapper_lib_features = rfa_idx
    filter_lib_features = X_train.columns[select_k_best.get_support(indices=True)]
    data = {
        "Embedded Method (RandomForest)": embedded_features.tolist(),
        "Wrapper Method (SFS)": wrapper_features,
        "Filter Method (Spearman)": filter_features,
        "Embedded Method (L1-Regularization)": l1_features.tolist(),
        "Wrapper Method (RFE)": wrapper_lib_features.tolist(),
        "Filter Method (Mutual Info)": filter_lib_features.tolist()
    }
    df = pd.DataFrame(data)
    df.to_csv("selected_features.csv", index=False)

    print("Embedded Method (RandomForest):", embedded_features.tolist())
    print("Wrapper Method (SFS):", wrapper_features.tolist())
    print("Filter Method (Spearman):", filter_features)
    print("Embedded Method (L1-Regularization):", l1_features.tolist())
    print("Wrapper Method (RFE):", wrapper_lib_features.tolist())
    print("Filter Method (Mutual Info):", filter_lib_features.tolist())

    return None


# data_methods(X_train, y_train, k_features)

selected_features_df = pd.read_csv("selected_features.csv")

def create_new_df(df, selected_features):
    return df[selected_features]

embedded_method_df = create_new_df(df_CV, selected_features_df["Embedded Method (RandomForest)"])
wrapper_method_df = create_new_df(df_CV, selected_features_df["Wrapper Method (SFS)"])
filter_method_df = create_new_df(df_CV, selected_features_df["Filter Method (Spearman)"])
embedded_l1_method_df = create_new_df(df_CV, selected_features_df["Embedded Method (L1-Regularization)"])
wrapper_lib_method_df = create_new_df(df_CV, selected_features_df["Wrapper Method (RFE)"])
filter_lib_method_df = create_new_df(df_CV, selected_features_df["Filter Method (Mutual Info)"])

models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "SVM": SVC()
}

results = {"Before": {}, "After": {}}

# До выбора признаков
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results["Before"][name] = accuracy_score(y_test, y_pred)

X_train_filter, X_test_filter, y_train_filter, y_test_filter = train_test_split(filter_method_df, y, test_size=0.80, random_state=43)
X_train_selected = X_train_filter
X_test_selected = X_test_filter

# После выбора признаков
for name, model in models.items():
    model.fit(X_train_selected, y_train)
    y_pred = model.predict(X_test_selected)
    results["After"][name] = accuracy_score(y_test, y_pred)

print("Before and after feature:")
print(pd.DataFrame(results))


# Кластеризация до выбора признаков
kmeans = KMeans(n_clusters=2, random_state=42)
clusters_before = kmeans.fit_predict(X_train)
ari_before = adjusted_rand_score(y_train, clusters_before)
silhouette_before = silhouette_score(X_train, clusters_before)

# Кластеризация после выбора признаков
clusters_after = kmeans.fit_predict(X_train_selected)
ari_after = adjusted_rand_score(y_train, clusters_after)
silhouette_after = silhouette_score(X_train_selected, clusters_after)

print("Clustering Quality:")
print(f"ARI Before: {ari_before}, After: {ari_after}")
print(f"Silhouette Before: {silhouette_before}, After: {silhouette_after}")


# PCA до и после выбора признаков
pca_before = PCA(n_components=2).fit_transform(X_train)
pca_after = PCA(n_components=2).fit_transform(X_train_selected)

# t-SNE до и после выбора признаков
tsne_before = TSNE(n_components=2, random_state=42).fit_transform(X_train)
tsne_after = TSNE(n_components=2, random_state=42).fit_transform(X_train_selected)


def plot_2D(data, labels, title):
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=10)
    plt.colorbar()
    plt.title(title)
    plt.show()

# PCA
plot_2D(pca_before, y_train, "PCA Before (Real Classes)")
plot_2D(pca_after, y_train, "PCA After (Real Classes)")

# t-SNE
plot_2D(tsne_before, y_train, "t-SNE Before (Real Classes)")
plot_2D(tsne_after, y_train, "t-SNE After (Real Classes)")

# Кластеры
plot_2D(pca_before, clusters_before, "PCA Before (Clusters)")
plot_2D(pca_after, clusters_after, "PCA After (Clusters)")

plot_2D(tsne_before, clusters_before, "t-SNE Before (Clusters)")
plot_2D(tsne_after, clusters_after, "t-SNE After (Clusters)")