# Шпаргалка для олимпиады по ИИ: Продвинутые модели и AutoML

## Содержание
1. [Градиентный бустинг](#градиентный-бустинг)
   - [XGBoost](#xgboost)
   - [LightGBM](#lightgbm)
   - [CatBoost](#catboost)
2. [AutoML-решения](#automl-решения)
   - [AutoML H2O](#automl-h2o)
   - [TPOT](#tpot)
   - [Auto-Sklearn](#auto-sklearn)
   - [PyTorch AutoML](#pytorch-automl)
3. [Продвинутые модели классификации и регрессии](#продвинутые-модели-классификации-и-регрессии)
   - [Стекинг](#стекинг)
   - [Блендинг](#блендинг)
   - [Нейросетевые модели](#нейросетевые-модели)

## Градиентный бустинг

Градиентный бустинг — это мощная техника машинного обучения, основанная на последовательном построении ансамбля слабых моделей (обычно деревьев решений). Каждая последующая модель обучается на ошибках предыдущих.

### XGBoost

**Теория:**

XGBoost (eXtreme Gradient Boosting) — высокоэффективная реализация градиентного бустинга, разработанная для скорости и производительности.

**Ключевые особенности:**
- Регуляризация (L1 и L2) для предотвращения переобучения
- Масштабируемость через параллельные вычисления
- Обработка разреженных данных
- Встроенная обработка отсутствующих значений
- Отсечение ветвей на ранних этапах (pruning)
- Возможность работы с кросс-валидацией

**Важные гиперпараметры:**
- `max_depth`: максимальная глубина дерева
- `learning_rate`: скорость обучения (eta)
- `n_estimators`: количество деревьев
- `subsample`: доля выборки для каждого дерева
- `colsample_bytree`: доля признаков для каждого дерева
- `gamma`: минимальное уменьшение потери для дальнейшего разделения
- `reg_alpha`: L1 регуляризация
- `reg_lambda`: L2 регуляризация

**Пример реализации с нуля (упрощенно):**

```python
import numpy as np

class SimpleXGBoostTree:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.tree = {}
    
    def _calculate_gain(self, y_left, y_right, y_parent):
        # Упрощенный расчет информационного выигрыша
        gain = np.sum(y_left)**2/len(y_left) + np.sum(y_right)**2/len(y_right) - np.sum(y_parent)**2/len(y_parent)
        return gain
    
    def _find_best_split(self, X, y):
        best_gain = 0
        best_feature = None
        best_threshold = None
        
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_idx = X[:, feature] <= threshold
                right_idx = ~left_idx
                
                if np.sum(left_idx) == 0 or np.sum(right_idx) == 0:
                    continue
                
                gain = self._calculate_gain(y[left_idx], y[right_idx], y)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _build_tree(self, X, y, depth=0):
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            return np.mean(y)
        
        feature, threshold, gain = self._find_best_split(X, y)
        
        if feature is None or gain <= 0:
            return np.mean(y)
        
        left_idx = X[:, feature] <= threshold
        right_idx = ~left_idx
        
        left_subtree = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right_subtree = self._build_tree(X[right_idx], y[right_idx], depth + 1)
        
        return {
            'feature': feature,
            'threshold': threshold,
            'left': left_subtree,
            'right': right_subtree
        }
    
    def _predict_sample(self, x, tree):
        if not isinstance(tree, dict):
            return tree
        
        if x[tree['feature']] <= tree['threshold']:
            return self._predict_sample(x, tree['left'])
        else:
            return self._predict_sample(x, tree['right'])
    
    def fit(self, X, y):
        self.tree = self._build_tree(X, y)
        return self
    
    def predict(self, X):
        return np.array([self._predict_sample(x, self.tree) for x in X])

class SimpleXGBoost:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.base_prediction = None
    
    def fit(self, X, y):
        # Инициализация с базовым предсказанием (среднее значение)
        self.base_prediction = np.mean(y)
        predictions = np.ones_like(y) * self.base_prediction
        
        for _ in range(self.n_estimators):
            # Вычисление градиентов (для регрессии с MSE это просто остатки)
            residuals = y - predictions
            
            # Построение дерева на остатках
            tree = SimpleXGBoostTree(max_depth=self.max_depth)
            tree.fit(X, residuals)
            
            # Обновление предсказаний
            update = tree.predict(X)
            predictions += self.learning_rate * update
            
            # Сохранение дерева
            self.trees.append(tree)
        
        return self
    
    def predict(self, X):
        predictions = np.ones(X.shape[0]) * self.base_prediction
        
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)
        
        return predictions
```

**Использование XGBoost из библиотеки:**

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import matplotlib.pyplot as plt

# Загрузка данных
data = load_boston()
X, y = data.data, data.target

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели XGBoost для регрессии
model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.01,
    reg_lambda=1.0,
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    eval_metric='rmse',
    early_stopping_rounds=10,
    verbose=False
)

# Предсказание
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse:.4f}")

# Визуализация важности признаков
plt.figure(figsize=(10, 6))
xgb.plot_importance(model, max_num_features=10)
plt.title('Feature Importance')
plt.show()

# Кросс-валидация с подбором гиперпараметров
param_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [50, 100, 200],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

grid_search = GridSearchCV(
    estimator=xgb.XGBRegressor(random_state=42),
    param_grid=param_grid,
    cv=3,
    scoring='neg_root_mean_squared_error',
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
print(f"Лучшие параметры: {grid_search.best_params_}")
print(f"Лучший RMSE: {-grid_search.best_score_:.4f}")

# Пример использования для классификации
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, roc_auc_score

# Загрузка данных для классификации
data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели XGBoost для классификации
model_clf = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

model_clf.fit(X_train, y_train)
y_pred_proba = model_clf.predict_proba(X_test)[:, 1]
y_pred = model_clf.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
```

**Обработка категориальных признаков:**

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

# Создаем синтетический датасет с категориальными признаками
data = {
    'category_1': np.random.choice(['A', 'B', 'C'], size=1000),
    'category_2': np.random.choice(['X', 'Y', 'Z'], size=1000),
    'numeric_1': np.random.normal(0, 1, size=1000),
    'numeric_2': np.random.normal(0, 1, size=1000),
    'target': np.random.randint(0, 2, size=1000)
}

df = pd.DataFrame(data)

# Кодирование категориальных признаков средствами pandas
df['category_1'] = df['category_1'].astype('category')
df['category_2'] = df['category_2'].astype('category')

# Подготовка данных
X = df[['category_1', 'category_2', 'numeric_1', 'numeric_2']]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание датасетов LightGBM с указанием категориальных признаков
categorical_features = ['category_1', 'category_2']
train_data = lgb.Dataset(
    X_train, 
    label=y_train, 
    categorical_feature=categorical_features
)

valid_data = lgb.Dataset(
    X_test, 
    label=y_test, 
    reference=train_data,
    categorical_feature=categorical_features
)

# Обучение модели
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'min_data_in_leaf': 20,
    'verbose': -1
}

model = lgb.train(
    params,
    train_data,
    num_boost_round=100,
    valid_sets=[valid_data],
    callbacks=[lgb.early_stopping(stopping_rounds=10)]
)

# Получение предсказаний
y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
y_pred = (y_pred_proba > 0.5).astype(int)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
```

### CatBoost

**Теория:**

CatBoost — реализация градиентного бустинга, разработанная Яндексом, особенно эффективная для работы с категориальными данными.

**Ключевые особенности:**
- Ordered Boosting для борьбы с переобучением
- Автоматическая обработка категориальных признаков
- Поддержка различных типов потерь
- Быстрая инференс-модель (предсказания)
- Работа с текстовыми признаками
- Поддержка GPU

**Важные гиперпараметры:**
- `iterations`: количество деревьев
- `learning_rate`: скорость обучения
- `depth`: глубина дерева
- `l2_leaf_reg`: коэффициент L2 регуляризации
- `random_seed`: начальное значение для генератора случайных чисел
- `bagging_temperature`: параметр для контроля случайности
- `border_count`: количество разбиений для числовых признаков
- `feature_border_type`: тип разбиения для числовых признаков

**Пример реализации с нуля (упрощенно):**

```python
import numpy as np
from collections import Counter

class SimpleCatBoostNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        
    def is_leaf(self):
        return self.value is not None

class SimpleCatBoostTree:
    def __init__(self, max_depth=6, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        
    def _ordered_target_statistics(self, feature_values, target_values):
        # Упрощенная реализация упорядоченного кодирования
        # В реальном CatBoost здесь используются более сложные алгоритмы
        value_to_targets = {}
        for value, target in zip(feature_values, target_values):
            if value not in value_to_targets:
                value_to_targets[value] = []
            value_to_targets[value].append(target)
        
        value_to_mean = {value: np.mean(targets) for value, targets in value_to_targets.items()}
        return value_to_mean
    
    def _find_best_split(self, X, y, depth):
        m, n = X.shape
        if m <= self.min_samples_split:
            return None, None, None
        
        best_gain = -float('inf')
        best_feature = None
        best_threshold = None
        
        for feature_idx in range(n):
            feature_values = X[:, feature_idx]
            
            # Проверка, является ли признак категориальным
            if len(np.unique(feature_values)) < 10:  # Примерная эвристика для определения категориального признака
                # Обработка категориального признака
                value_to_mean = self._ordered_target_statistics(feature_values, y)
                
                # Преобразуем категории в числовые значения на основе средних целевых значений
                transformed_values = np.array([value_to_mean[val] for val in feature_values])
                unique_values = np.unique(transformed_values)
            else:
                # Числовой признак
                transformed_values = feature_values
                unique_values = np.unique(transformed_values)
            
            # Ищем лучший порог для разбиения
            for threshold in unique_values:
                left_indices = transformed_values <= threshold
                right_indices = ~left_indices
                
                if np.sum(left_indices) < self.min_samples_split or np.sum(right_indices) < self.min_samples_split:
                    continue
                
                # Расчет информационного выигрыша
                y_left = y[left_indices]
                y_right = y[right_indices]
                
                # Энтропия или дисперсия в зависимости от задачи
                parent_var = np.var(y) * len(y)
                left_var = np.var(y_left) * len(y_left)
                right_var = np.var(y_right) * len(y_right)
                
                gain = parent_var - (left_var + right_var)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _build_tree(self, X, y, depth=0):
        m, n = X.shape
        
        # Условия остановки
        if depth >= self.max_depth or m <= self.min_samples_split or np.all(y == y[0]):
            leaf_value = np.mean(y)
            return SimpleCatBoostNode(value=leaf_value)
        
        # Поиск лучшего разбиения
        best_feature, best_threshold, best_gain = self._find_best_split(X, y, depth)
        
        if best_feature is None or best_gain <= 0:
            leaf_value = np.mean(y)
            return SimpleCatBoostNode(value=leaf_value)
        
        # Создание потомков
        feature_values = X[:, best_feature]
        
        # Проверка, является ли признак категориальным
        if len(np.unique(feature_values)) < 10:  # Эвристика
            value_to_mean = self._ordered_target_statistics(feature_values, y)
            transformed_values = np.array([value_to_mean[val] for val in feature_values])
            left_indices = transformed_values <= best_threshold
        else:
            left_indices = feature_values <= best_threshold
            
        right_indices = ~left_indices
        
        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)
        
        return SimpleCatBoostNode(
            feature=best_feature,
            threshold=best_threshold,
            left=left_subtree,
            right=right_subtree
        )
    
    def fit(self, X, y):
        self.root = self._build_tree(X, y)
        return self
    
    def _predict_sample(self, x, node):
        if node.is_leaf():
            return node.value
        
        # Преобразование для категориальных признаков в реальной модели будет сложнее
        if x[node.feature] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)
    
    def predict(self, X):
        return np.array([self._predict_sample(x, self.root) for x in X])

class SimpleCatBoost:
    def __init__(self, iterations=100, learning_rate=0.1, depth=6, min_samples_split=2):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.min_samples_split = min_samples_split
        self.trees = []
        self.base_prediction = None
    
    def fit(self, X, y):
        self.base_prediction = np.mean(y)
        predictions = np.ones_like(y) * self.base_prediction
        
        for _ in range(self.iterations):
            residuals = y - predictions
            
            tree = SimpleCatBoostTree(max_depth=self.depth, min_samples_split=self.min_samples_split)
            tree.fit(X, residuals)
            
            update = tree.predict(X)
            predictions += self.learning_rate * update
            
            self.trees.append(tree)
        
        return self
    
    def predict(self, X):
        predictions = np.ones(X.shape[0]) * self.base_prediction
        
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)
        
        return predictions
```

**Использование CatBoost из библиотеки:**

```python
import numpy as np
import pandas as pd
from catboost import CatBoost, CatBoostRegressor, CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score
from sklearn.datasets import load_boston, load_breast_cancer
import matplotlib.pyplot as plt

# Пример для регрессии
# Загрузка данных
data = load_boston()
X, y = data.data, data.target

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели CatBoost для регрессии
model = CatBoostRegressor(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    loss_function='RMSE',
    eval_metric='RMSE',
    random_seed=42,
    verbose=0
)

model.fit(
    X_train, y_train,
    eval_set=(X_test, y_test),
    early_stopping_rounds=50
)

# Предсказание
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse:.4f}")

# Визуализация важности признаков
plt.figure(figsize=(10, 6))
feature_names = [f'feature_{i}' for i in range(X.shape[1])]
feature_importance = model.get_feature_importance()
plt.barh(range(len(feature_names)), feature_importance, align='center')
plt.yticks(range(len(feature_names)), feature_names)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()

# Подбор гиперпараметров
param_grid = {
    'depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'iterations': [100, 200, 300],
    'l2_leaf_reg': [1, 3, 5, 7],
    'border_count': [32, 64, 128]
}

grid_search = GridSearchCV(
    estimator=CatBoostRegressor(
        loss_function='RMSE',
        eval_metric='RMSE',
        random_seed=42,
        verbose=0
    ),
    param_grid=param_grid,
    cv=3,
    scoring='neg_root_mean_squared_error',
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
print(f"Лучшие параметры: {grid_search.best_params_}")
print(f"Лучший RMSE: {-grid_search.best_score_:.4f}")

# Пример для классификации
# Загрузка данных
data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели CatBoost для классификации
model_clf = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    loss_function='Logloss',
    eval_metric='AUC',
    random_seed=42,
    verbose=0
)

model_clf.fit(
    X_train, y_train,
    eval_set=(X_test, y_test),
    early_stopping_rounds=50
)

# Предсказание
y_pred_proba = model_clf.predict_proba(X_test)[:, 1]
y_pred = model_clf.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
```

**Работа с категориальными признаками в CatBoost:**

```python
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# Создаем синтетический датасет с категориальными признаками
data = {
    'category_1': np.random.choice(['A', 'B', 'C'], size=1000),
    'category_2': np.random.choice(['X', 'Y', 'Z'], size=1000),
    'numeric_1': np.random.normal(0, 1, size=1000),
    'numeric_2': np.random.normal(0, 1, size=1000),
    'target': np.random.randint(0, 2, size=1000)
}

df = pd.DataFrame(data)

# Разделение на обучающую и тестовую выборки
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Указание категориальных признаков
cat_features = ['category_1', 'category_2']

# Создание пулов данных CatBoost
train_pool = Pool(X_train, y_train, cat_features=cat_features)
test_pool = Pool(X_test, y_test, cat_features=cat_features)

# Обучение модели
model = CatBoostClassifier(
    iterations=300,
    learning_rate=0.05,
    depth=6,
    loss_function='Logloss',
    eval_metric='AUC',
    verbose=0,
    random_seed=42
)

model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=50)

# Предсказание
y_pred_proba = model.predict_proba(test_pool)[:, 1]
y_pred = model.predict(test_pool)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")

# Визуализация дерева
first_tree = model.get_tree(0)
print(first_tree)

# Текстовые признаки в CatBoost
# Создаем датасет с текстовыми данными
text_data = {
    'text': [
        'Это пример текста для классификации',
        'Машинное обучение - интересная тема',
        'CatBoost хорошо работает с текстом',
        'Глубокое обучение требует много данных',
        'Анализ текста важен для NLP'
    ] * 20,
    'category': np.random.choice(['Tech', 'Science', 'Art'], size=100),
    'target': np.random.randint(0, 2, size=100)
}

text_df = pd.DataFrame(text_data)

# Разделение на обучающую и тестовую выборки
X_text = text_df.drop('target', axis=1)
y_text = text_df['target']

X_train_text, X_test_text, y_train_text, y_test_text = train_test_split(X_text, y_text, test_size=0.2, random_state=42)

# Указание текстовых и категориальных признаков
cat_features = ['category']
text_features = ['text']

# Создание пулов данных CatBoost с текстовыми признаками
train_pool_text = Pool(
    X_train_text, 
    y_train_text, 
    cat_features=cat_features,
    text_features=text_features
)

test_pool_text = Pool(
    X_test_text, 
    y_test_text, 
    cat_features=cat_features,
    text_features=text_features
)

# Обучение модели с текстовыми признаками
model_text = CatBoostClassifier(
    iterations=100,
    learning_rate=0.1,
    depth=6,
    loss_function='Logloss',
    text_processing={
        'tokenizers': [{
            'tokenizer_id': 'Space',
        }],
        'dictionaries': [{
            'dictionary_id': 'BiGram',
            'max_dictionary_size': 50000,
            'gram_order': 2,
        }],
        'feature_calculators': [{
            'feature_calcualtor_id': 'BoW',
            'dictionary_id': 'BiGram',
            'tokenizer_id': 'Space',
        }]
    },
    verbose=0
)

model_text.fit(train_pool_text, eval_set=test_pool_text)

# Предсказание
y_pred_text = model_text.predict(test_pool_text)
print(f"Accuracy with text features: {accuracy_score(y_test_text, y_pred_text):.4f}")
```

## AutoML-решения

AutoML (Automated Machine Learning) — это технологии, которые автоматизируют процесс создания моделей машинного обучения, включая подбор алгоритмов, признаков и гиперпараметров.

### AutoML H2O

**Теория:**

H2O AutoML — это инструмент для автоматического создания и оптимизации моделей машинного обучения. Он создает ансамбль из различных моделей и выбирает лучшую на основе заданной метрики.

**Ключевые особенности:**
- Автоматический подбор и обучение моделей
- Встроенная кросс-валидация
- Создание стекинг-ансамблей
- Поддержка распределенных вычислений
- Обработка данных большого объема
- Прозрачность процесса обучения

**Пример использования H2O AutoML:**

```python
import h2o
from h2o.automl import H2OAutoML
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston, load_breast_cancer
from sklearn.model_selection import train_test_split

# Инициализация H2O
h2o.init()

# Пример для регрессии
# Загрузка данных
data = load_boston()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Разделение на обучающую и тестовую выборки
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Преобразование в формат H2O
train_h2o = h2o.H2OFrame(train_df)
test_h2o = h2o.H2OFrame(test_df)

# Определение предикторов и целевой переменной
y = 'target'
x = train_h2o.columns
x.remove(y)

# Обучение AutoML
aml = H2OAutoML(
    max_models=20,
    seed=42,
    max_runtime_secs=300,  # 5 минут максимум
    sort_metric="RMSE"
)

aml.train(x=x, y=y, training_frame=train_h2o, validation_frame=test_h2o)

# Получение лучшей модели
best_model = aml.leader

# Вывод информации о модели
print(best_model)

# Список всех моделей, отсортированных по производительности
lb = aml.leaderboard
print(lb)

# Предсказание
predictions = best_model.predict(test_h2o)
print(predictions.head())

# Оценка производительности
perf = best_model.model_performance(test_h2o)
print(perf)

# Пример для классификации
# Загрузка данных
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Разделение на обучающую и тестовую выборки
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Преобразование в формат H2O
train_h2o = h2o.H2OFrame(train_df)
test_h2o = h2o.H2OFrame(test_df)

# Определение предикторов и целевой переменной
y = 'target'
x = train_h2o.columns
x.remove(y)

# Указываем, что целевая переменная является категориальной для задачи классификации
train_h2o[y] = train_h2o[y].asfactor()
test_h2o[y] = test_h2o[y].asfactor()

# Обучение AutoML для классификации
aml_clf = H2OAutoML(
    max_models=20,
    seed=42,
    max_runtime_secs=300,
    sort_metric="AUC"
)

aml_clf.train(x=x, y=y, training_frame=train_h2o, validation_frame=test_h2o)

# Получение лучшей модели
best_model_clf = aml_clf.leader

# Вывод информации о модели
print(best_model_clf)

# Предсказание
predictions_clf = best_model_clf.predict(test_h2o)
print(predictions_clf.head())

# Оценка производительности
perf_clf = best_model_clf.model_performance(test_h2o)
print(perf_clf)

# Сохранение и загрузка модели
model_path = h2o.save_model(model=best_model_clf, path="./h2o_model", force=True)
loaded_model = h2o.load_model(model_path)

# Экспорт моделей в формат MOJO для быстрого развертывания
mojo_path = best_model_clf.download_mojo(path="./", get_genmodel_jar=True)

# Завершение сессии H2O
h2o.shutdown()
```

**Продвинутые возможности H2O AutoML:**

```python
import h2o
from h2o.automl import H2OAutoML
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Инициализация H2O
h2o.init()

# Загрузка данных
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Добавление категориальных признаков для демонстрации
df['cat_feature'] = np.random.choice(['A', 'B', 'C', 'D'], size=len(df))

# Разделение на обучающую и тестовую выборки
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Преобразование в формат H2O
train_h2o = h2o.H2OFrame(train_df)
test_h2o = h2o.H2OFrame(test_df)

# Определение предикторов и целевой переменной
y = 'target'
x = train_h2o.columns
x.remove(y)

# Указание категориальных признаков
train_h2o['cat_feature'] = train_h2o['cat_feature'].asfactor()
test_h2o['cat_feature'] = test_h2o['cat_feature'].asfactor()

# Обучение AutoML с расширенными параметрами
aml = H2OAutoML(
    max_models=20,
    seed=42,
    max_runtime_secs=600,
    sort_metric="RMSE",
    include_algos=["GBM", "DRF", "XGBoost", "GLM"],  # Алгоритмы для включения
    exclude_algos=["DeepLearning"],  # Алгоритмы для исключения
    balance_classes=False,  # Балансировка классов для классификации
    keep_cross_validation_predictions=True,  # Сохранять предсказания для создания стекинг-ансамбля
    keep_cross_validation_models=True,  # Сохранять модели кросс-валидации
    nfolds=5,  # Количество фолдов для кросс-валидации
    verbosity="info"  # Уровень вывода информации
)

aml.train(x=x, y=y, training_frame=train_h2o, validation_frame=test_h2o)

# Получение лучшей модели
best_model = aml.leader

# Все модели, обученные AutoML
all_models = aml.leaderboard.as_data_frame()
print(all_models)

# Извлечение информации о параметрах лучшей модели
params = best_model.params
print(params)

# Объяснение модели с помощью SHAP-значений (для GBM и XGBoost)
if best_model.algo in ["gbm", "xgboost"]:
    # Создание объекта для интерпретации модели
    from h2o.estimators.gbm import H2OGradientBoostingEstimator
    h2o_model = H2OGradientBoostingEstimator(model_id=best_model.model_id)
    contrib = h2o_model.predict_contributions(test_h2o)
    print(contrib.head())

# Частичная зависимость (для анализа влияния признаков)
pdp = best_model.partial_plot(train_h2o, cols=["MedInc", "HouseAge"], plot=True)

# Создание и оценка ансамбля (стекинг)
from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator

# Получение базовых моделей
base_models = aml.leaderboard['model_id'].as_data_frame()['model_id'].tolist()[:5]  # Топ-5 моделей

# Создание ансамбля
ensemble = H2OStackedEnsembleEstimator(
    base_models=base_models,
    seed=42
)

ensemble.train(x=x, y=y, training_frame=train_h2o, validation_frame=test_h2o)

# Оценка ансамбля
ensemble_perf = ensemble.model_performance(test_h2o)
print(ensemble_perf)

# Сравнение с лучшей моделью
best_perf = best_model.model_performance(test_h2o)
print(f"Лучшая модель RMSE: {best_perf.rmse()}")
print(f"Ансамбль RMSE: {ensemble_perf.rmse()}")

# Экспорт моделей для развертывания
h2o.save_model(model=best_model, path="./best_model", force=True)
h2o.save1000),
    'target': np.random.randint(0, 2, size=1000)
}

df = pd.DataFrame(data)

# Кодирование категориальных признаков
le_1 = LabelEncoder()
le_2 = LabelEncoder()

df['category_1_encoded'] = le_1.fit_transform(df['category_1'])
df['category_2_encoded'] = le_2.fit_transform(df['category_2'])

# Подготовка данных
X = df[['category_1_encoded', 'category_2_encoded', 'numeric_1', 'numeric_2']]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Альтернативный метод: использование DMatrix с указанием категориальных признаков
dtrain = xgb.DMatrix(X_train, y_train, feature_names=X.columns, 
                    enable_categorical=True, 
                    feature_types=['c', 'c', 'q', 'q'])  # c - категориальные, q - количественные
dtest = xgb.DMatrix(X_test, y_test, feature_names=X.columns, 
                   enable_categorical=True, 
                   feature_types=['c', 'c', 'q', 'q'])

# Обучение модели
params = {
    'objective': 'binary:logistic',
    'max_depth': 3,
    'learning_rate': 0.1,
    'tree_method': 'hist'  # Быстрый алгоритм для категориальных признаков
}

model = xgb.train(
    params,
    dtrain,
    num_boost_round=100,
    evals=[(dtrain, 'train'), (dtest, 'eval')],
    early_stopping_rounds=10,
    verbose_eval=False
)

# Получение предсказаний
y_pred_proba = model.predict(dtest)
y_pred = (y_pred_proba > 0.5).astype(int)

# Оценка производительности
from sklearn.metrics import accuracy_score, roc_auc_score
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
```

### LightGBM

**Теория:**

LightGBM (Light Gradient Boosting Machine) — эффективная реализация градиентного бустинга, разработанная Microsoft, оптимизированная для скорости и памяти.

**Ключевые особенности:**
- Построение дерева по листьям (Leaf-wise), а не по уровням (Level-wise)
- GOSS (Gradient-based One-Side Sampling) для эффективного отбора примеров
- EFB (Exclusive Feature Bundling) для сокращения размерности
- Параллельное обучение
- Непосредственная поддержка категориальных признаков
- Оптимизация для данных большой размерности

**Важные гиперпараметры:**
- `num_leaves`: количество листьев в дереве (вместо max_depth)
- `learning_rate`: скорость обучения
- `n_estimators`: количество итераций (деревьев)
- `max_depth`: максимальная глубина дерева
- `min_data_in_leaf`: минимальное количество объектов в листе
- `feature_fraction`: доля признаков для каждого дерева
- `bagging_fraction`: доля данных для каждого дерева
- `lambda_l1`: L1 регуляризация
- `lambda_l2`: L2 регуляризация

**Пример реализации с нуля (упрощенно):**

```python
import numpy as np

class SimpleLGBMNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        
    def is_leaf(self):
        return self.value is not None

class SimpleLGBMTree:
    def __init__(self, max_leaves=31, min_data_in_leaf=20):
        self.max_leaves = max_leaves
        self.min_data_in_leaf = min_data_in_leaf
        self.root = None
        
    def _compute_leaf_value(self, targets):
        # Для регрессии - среднее значение
        return np.mean(targets)
    
    def _find_best_split_leaf_wise(self, X, y, active_leaves):
        best_gain = -float('inf')
        best_leaf = None
        best_feature = None
        best_threshold = None
        best_left_indices = None
        best_right_indices = None
        
        for leaf_id in active_leaves:
            leaf_indices = active_leaves[leaf_id]
            
            if len(leaf_indices) < 2 * self.min_data_in_leaf:
                continue
                
            X_leaf = X[leaf_indices]
            y_leaf = y[leaf_indices]
            
            for feature in range(X_leaf.shape[1]):
                # Уникальные значения признака
                unique_values = np.unique(X_leaf[:, feature])
                if len(unique_values) <= 1:
                    continue
                
                # Потенциальные пороги для разделения
                thresholds = (unique_values[:-1] + unique_values[1:]) / 2
                
                for threshold in thresholds:
                    left_indices_local = X_leaf[:, feature] <= threshold
                    right_indices_local = ~left_indices_local
                    
                    if np.sum(left_indices_local) < self.min_data_in_leaf or np.sum(right_indices_local) < self.min_data_in_leaf:
                        continue
                    
                    left_values = y_leaf[left_indices_local]
                    right_values = y_leaf[right_indices_local]
                    
                    # Упрощенный расчет прироста информации
                    parent_var = np.var(y_leaf) * len(y_leaf)
                    left_var = np.var(left_values) * len(left_values)
                    right_var = np.var(right_values) * len(right_values)
                    
                    gain = parent_var - (left_var + right_var)
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_leaf = leaf_id
                        best_feature = feature
                        best_threshold = threshold
                        
                        # Сохраняем глобальные индексы
                        left_indices_global = np.array(leaf_indices)[left_indices_local]
                        right_indices_global = np.array(leaf_indices)[right_indices_local]
                        best_left_indices = left_indices_global
                        best_right_indices = right_indices_global
        
        return best_leaf, best_feature, best_threshold, best_left_indices, best_right_indices, best_gain
    
    def fit(self, X, y):
        if len(np.unique(y)) == 1:
            self.root = SimpleLGBMNode(value=self._compute_leaf_value(y))
            return self
        
        # Словарь активных листьев
        active_leaves = {0: np.arange(len(y))}
        leaf_nodes = {}
        leaf_counter = 0
        
        while len(active_leaves) > 0 and leaf_counter < self.max_leaves:
            best_leaf, feature, threshold, left_indices, right_indices, gain = self._find_best_split_leaf_wise(X, y, active_leaves)
            
            if best_leaf is None or gain <= 0:
                # Если нет хорошего разделения, превращаем все активные листья в листовые узлы
                for leaf_id, indices in active_leaves.items():
                    leaf_nodes[leaf_id] = SimpleLGBMNode(value=self._compute_leaf_value(y[indices]))
                break
            
            # Удаляем лучший лист из активных
            indices = active_leaves.pop(best_leaf)
            
            # Создаем новые листья
            left_leaf_id = leaf_counter + 1
            right_leaf_id = leaf_counter + 2
            leaf_counter += 2
            
            # Добавляем новые листья в активные
            active_leaves[left_leaf_id] = left_indices
            active_leaves[right_leaf_id] = right_indices
            
            # Создаем узел для лучшего листа
            leaf_nodes[best_leaf] = SimpleLGBMNode(
                feature=feature,
                threshold=threshold,
                left=left_leaf_id,
                right=right_leaf_id
            )
            
            # Если достигли максимального количества листьев
            if leaf_counter >= self.max_leaves - 1:
                # Преобразуем все оставшиеся активные листья в листовые узлы
                for leaf_id, indices in active_leaves.items():
                    leaf_nodes[leaf_id] = SimpleLGBMNode(value=self._compute_leaf_value(y[indices]))
                break
        
        # Строим окончательное дерево, заменяя id на узлы
        def build_tree(leaf_id):
            node = leaf_nodes[leaf_id]
            if not node.is_leaf():
                node.left = build_tree(node.left)
                node.right = build_tree(node.right)
            return node
        
        self.root = build_tree(0)
        return self
    
    def _predict_sample(self, x, node):
        if node.is_leaf():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)
    
    def predict(self, X):
        return np.array([self._predict_sample(x, self.root) for x in X])

class SimpleLGBM:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_leaves=31, min_data_in_leaf=20):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_leaves = max_leaves
        self.min_data_in_leaf = min_data_in_leaf
        self.trees = []
        self.base_prediction = None
    
    def fit(self, X, y):
        # Инициализация с базовым предсказанием (среднее значение)
        self.base_prediction = np.mean(y)
        predictions = np.ones_like(y) * self.base_prediction
        
        for _ in range(self.n_estimators):
            # Вычисление градиентов (для регрессии с MSE это просто остатки)
            residuals = y - predictions
            
            # Построение дерева на остатках
            tree = SimpleLGBMTree(max_leaves=self.max_leaves, min_data_in_leaf=self.min_data_in_leaf)
            tree.fit(X, residuals)
            
            # Обновление предсказаний
            update = tree.predict(X)
            predictions += self.learning_rate * update
            
            # Сохранение дерева
            self.trees.append(tree)
        
        return self
    
    def predict(self, X):
        predictions = np.ones(X.shape[0]) * self.base_prediction
        
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)
        
        return predictions
```

**Использование LightGBM из библиотеки:**

```python
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score
from sklearn.datasets import load_boston, load_breast_cancer
import matplotlib.pyplot as plt

# Пример для регрессии
# Загрузка данных
data = load_boston()
X, y = data.data, data.target

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели LightGBM для регрессии
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1
}

# Создание датасетов LightGBM
train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# Обучение модели
model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[valid_data],
    callbacks=[lgb.early_stopping(stopping_rounds=100)]
)

# Предсказание
y_pred = model.predict(X_test, num_iteration=model.best_iteration)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse:.4f}")

# Визуализация важности признаков
plt.figure(figsize=(10, 6))
lgb.plot_importance(model, max_num_features=10)
plt.title('Feature Importance')
plt.show()

# Использование LightGBM через API scikit-learn
model_sklearn = lgb.LGBMRegressor(
    objective='regression',
    num_leaves=31,
    learning_rate=0.05,
    n_estimators=100,
    feature_fraction=0.9,
    bagging_fraction=0.8,
    bagging_freq=5,
    random_state=42
)

model_sklearn.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric='rmse',
    early_stopping_rounds=100,
    verbose=False
)

y_pred_sklearn = model_sklearn.predict(X_test)
rmse_sklearn = np.sqrt(mean_squared_error(y_test, y_pred_sklearn))
print(f"RMSE (sklearn API): {rmse_sklearn:.4f}")

# Подбор гиперпараметров
param_grid = {
    'num_leaves': [15, 31, 63],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300],
    'min_child_samples': [5, 10, 20],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

grid_search = GridSearchCV(
    estimator=lgb.LGBMRegressor(random_state=42),
    param_grid=param_grid,
    cv=3,
    scoring='neg_root_mean_squared_error',
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
print(f"Лучшие параметры: {grid_search.best_params_}")
print(f"Лучший RMSE: {-grid_search.best_score_:.4f}")

# Пример для классификации
# Загрузка данных
data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели LightGBM для классификации
params_clf = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1
}

train_data_clf = lgb.Dataset(X_train, label=y_train)
valid_data_clf = lgb.Dataset(X_test, label=y_test, reference=train_data_clf)

model_clf = lgb.train(
    params_clf,
    train_data_clf,
    num_boost_round=1000,
    valid_sets=[valid_data_clf],
    callbacks=[lgb.early_stopping(stopping_rounds=100)]
)

# Предсказание
y_pred_proba = model_clf.predict(X_test, num_iteration=model_clf.best_iteration)
y_pred = (y_pred_proba > 0.5).astype(int)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
```

**Работа с категориальными признаками в LightGBM:**

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score

# Создаем синтетический датасет с категориальными признаками
data = {
    'category_1': np.random.choice(['A', 'B', 'C'], size=1000),
    'category_2': np.random.choice(['X', 'Y', 'Z'], size=1000),
    'numeric_1': np.random.normal(0, 1, size=1000),
    'numeric_2': np.random.normal(0, 1, size=