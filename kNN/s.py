import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Загружаем данные (пример в pandas DataFrame)
data = pd.read_csv("../DataSet/players_data.csv")  # замените на фактический путь

data = data.drop(columns=["Name", "Favorite Champions"])

y = data['Rank']
X = data.drop(columns=['Rank'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Пайплайн для предобработки данных
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Games Played', 'Win Rate %', 'Level', 'Kills', 'Deaths', 'Assists']),
        ('cat', OneHotEncoder(), ['Average Enemy Rating', 'Favorite Role'])
    ])

# Объединяем предобработку и модель в один пайплайн
knn_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', KNeighborsClassifier(n_neighbors=20))])

# Обучение модели
knn_pipeline.fit(X_train, y_train)

# Предсказание и оценка
y_pred = knn_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
