"""
Модуль с реализацией Decision Tree Regressor и Random Forest Regressor с нуля.

Содержит:
    - mse: функция вычисления среднеквадратичной ошибки.
    - NodeReg: класс узла дерева решений для регрессии.
    - DecisionTreeRegressor: класс дерева решений для регрессии.
    - CustomRandomForest: класс случайного леса для регрессии.
"""

import random
import math
from typing import List, Optional, Tuple


def mse(y: List[float]) -> float:
    """
    Вычисляет среднеквадратичную ошибку (Mean Squared Error) для списка значений.

    MSE измеряет среднее квадратов отклонений значений от их среднего.
    Используется как критерий информативности при построении дерева:
    чем меньше MSE, тем однороднее узел.

    Args:
        y: Список числовых значений целевой переменной.

    Returns:
        Среднеквадратичная ошибка. Возвращает 0.0, если список пуст.
    """
    if not y:
        return 0.0
    mean = sum(y) / len(y)
    return sum((yi - mean) ** 2 for yi in y) / len(y)


class NodeReg:
    """
    Узел дерева решений для задачи регрессии.

    Каждый узел может быть либо:
        - Внутренним (non-leaf): содержит индекс признака (feature_index),
          пороговое значение (threshold) и ссылки на левое/правое поддерево.
        - Листовым (leaf): содержит предсказанное значение (value).

    Attributes:
        feature_index: Индекс признака, по которому производится разбиение.
                       None для листового узла.
        threshold: Пороговое значение для разбиения. None для листового узла.
        left: Левое поддерево (значения <= threshold). None для листа.
        right: Правое поддерево (значения > threshold). None для листа.
        value: Предсказанное значение для листового узла. None для внутреннего узла.
    """

    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        """
        Инициализирует узел дерева.

        Args:
            feature_index: Индекс признака для разбиения.
            threshold: Пороговое значение разбиения.
            left: Левый дочерний узел.
            right: Правый дочерний узел.
            value: Предсказанное значение (для листа).
        """
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self) -> bool:
        """
        Проверяет, является ли узел листовым.

        Returns:
            True, если узел содержит предсказанное значение (value is not None).
        """
        return self.value is not None


class DecisionTreeRegressor:
    """
    Дерево решений для задачи регрессии.

    Алгоритм строит бинарное дерево, рекурсивно разбивая данные
    по признакам и порогам так, чтобы минимизировать взвешенную MSE.
    В листьях хранится среднее значение целевой переменной.

    Attributes:
        max_depth: Максимальная глубина дерева.
        min_samples_split: Минимальное количество образцов для разбиения узла.
        max_features: Количество признаков для рассмотрения при каждом разбиении.
                      Может быть int, float, 'sqrt', 'log2' или None (все признаки).
        random_state: Seed для воспроизводимости случайного выбора признаков.
        rng: Генератор случайных чисел (random.Random).
        root: Корневой узел дерева (NodeReg).
        n_features: Общее количество признаков в данных.
        feature_indices: Индексы признаков, использованные для построения дерева.
    """

    def __init__(self, max_depth: int = 5, min_samples_split: int = 2,
                 max_features=None, random_state: Optional[int] = None):
        """
        Инициализирует регрессор дерева решений.

        Args:
            max_depth: Максимальная глубина дерева. По умолчанию 5.
            min_samples_split: Минимальное число образцов для разбиения. По умолчанию 2.
            max_features: Число признаков для случайного подмножества.
                          int — фиксированное число,
                          float — доля от общего числа признаков,
                          'sqrt' — квадратный корень,
                          'log2' — логарифм по основанию 2,
                          None — все признаки.
            random_state: Seed для воспроизводимости.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.rng = random.Random(random_state)
        self.root = None
        self.n_features = None
        self.feature_indices = None

    def _get_max_features(self, n_features: int) -> int:
        """
        Преобразует параметр max_features в конкретное число признаков.

        Логика преобразования:
            - None → все признаки (n_features).
            - int → min(max_features, n_features).
            - float → max(1, int(max_features * n_features)).
            - 'sqrt' → max(1, int(sqrt(n_features))).
            - 'log2' → max(1, int(log2(n_features))).

        Args:
            n_features: Общее количество признаков в датасете.

        Returns:
            Целое число — количество признаков для рассмотрения при разбиении.
        """
        if self.max_features is None:
            return n_features
        if isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        if isinstance(self.max_features, float):
            return max(1, int(self.max_features * n_features))
        if self.max_features == 'sqrt':
            return max(1, int(math.sqrt(n_features)))
        if self.max_features == 'log2':
            return max(1, int(math.log2(n_features)))
        return n_features

    def _best_split(self, X: List[List[float]], y: List[float],
                    feature_indices: List[int]) -> Tuple[Optional[int], Optional[float]]:
        """
        Находит оптимальное разбиение для текущего узла.

        Для каждого указанного признака данные сортируются по значению признака.
        Затем перебираются все возможные пороги (среднее между соседними
        уникальными значениями) и выбирается тот, который минимизирует
        взвешенную сумму MSE левой и правой частей.

        Args:
            X: Матрица признаков (список списков).
            y: Вектор целевых значений.
            feature_indices: Индексы признаков, среди которых ищется лучшее разбиение.

        Returns:
            Кортеж (best_feature, best_threshold), где:
                best_feature — индекс лучшего признака или None,
                best_threshold — лучший порог или None.
            Если улучшение невозможно, возвращает (None, None).
        """
        best_mse = float('inf')
        best_feature = None
        best_threshold = None
        n_samples = len(y)

        # Перебираем только указанное подмножество признаков
        for feature_idx in feature_indices:
            # Сортируем индексы образцов по значению текущего признака
            sorted_indices = sorted(range(n_samples), key=lambda i: X[i][feature_idx])
            sorted_values = [X[i][feature_idx] for i in sorted_indices]
            sorted_targets = [y[i] for i in sorted_indices]

            # Перебираем возможные точки разбиения между соседними элементами
            for i in range(n_samples - 1):
                # Пропускаем разбиения, где целевые значения с обеих сторон одинаковы
                # (такое разбиение не даст улучшения)
                if sorted_targets[i] == sorted_targets[i + 1]:
                    continue

                # Порог — среднее между двумя соседними значениями признака
                threshold = (sorted_values[i] + sorted_values[i + 1]) / 2.0

                # Разделяем целевые значения на левую и правую части
                left_targets = sorted_targets[:i + 1]
                right_targets = sorted_targets[i + 1:]
                n_left, n_right = len(left_targets), len(right_targets)

                # Взвешенная сумма MSE: учитываем размеры подмножеств
                weighted_mse = (n_left / n_samples) * mse(left_targets) +                                (n_right / n_samples) * mse(right_targets)

                # Сохраняем разбиение с наименьшей ошибкой
                if weighted_mse < best_mse:
                    best_mse = weighted_mse
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold

    def _build_tree(self, X: List[List[float]], y: List[float],
                    depth: int = 0, feature_indices: Optional[List[int]] = None) -> NodeReg:
        """
        Рекурсивно строит дерево решений.

        Базовые случаи остановки (создаём лист):
            - Достигнута максимальная глубина (depth >= max_depth).
            - Слишком мало образцов для разбиения (n_samples < min_samples_split).
            - Нет образцов (n_samples == 0).
            - Не удалось найти подходящее разбиение (feature_idx is None).
            - Разбиение создало пустое поддерево.

        В листе сохраняется среднее значение целевой переменной.

        Args:
            X: Матрица признаков текущего подмножества.
            y: Целевые значения текущего подмножества.
            depth: Текущая глубина рекурсии.
            feature_indices: Индексы признаков для рассмотрения.

        Returns:
            Корневой узел (NodeReg) построенного (под)дерева.
        """
        n_samples = len(X)

        # --- Условия остановки: создаём листовой узел ---
        if (depth >= self.max_depth or
                n_samples < self.min_samples_split or
                n_samples == 0):
            return NodeReg(value=sum(y) / len(y) if y else 0)

        # Ищем лучшее разбиение
        feature_idx, threshold = self._best_split(X, y, feature_indices)

        # Если разбиение не найдено — лист
        if feature_idx is None:
            return NodeReg(value=sum(y) / len(y) if y else 0)

        # --- Разделяем данные на левое и правое поддеревья ---
        X_left, y_left, X_right, y_right = [], [], [], []
        for i in range(len(X)):
            if X[i][feature_idx] <= threshold:
                X_left.append(X[i])
                y_left.append(y[i])
            else:
                X_right.append(X[i])
                y_right.append(y[i])

        # Если одно из поддеревьев пустое — невозможно разделить, делаем лист
        if not y_left or not y_right:
            return NodeReg(value=sum(y) / len(y) if y else 0)

        # --- Рекурсивно строим поддеревья ---
        left_child = self._build_tree(X_left, y_left, depth + 1, feature_indices)
        right_child = self._build_tree(X_right, y_right, depth + 1, feature_indices)

        return NodeReg(
            feature_index=feature_idx,
            threshold=threshold,
            left=left_child,
            right=right_child
        )

    def fit(self, X: List[List[float]], y: List[float],
            feature_indices: Optional[List[int]] = None) -> 'DecisionTreeRegressor':
        """
        Обучает дерево решений на предоставленных данных.

        Если feature_indices не заданы, случайным образом выбирается
        подмножество признаков согласно параметру max_features.

        Args:
            X: Матрица признаков (n_samples x n_features).
            y: Вектор целевых значений (n_samples,).
            feature_indices: Явно заданные индексы признаков для построения.
                             Если None, выбираются случайно.

        Returns:
            self — обученный экземпляр DecisionTreeRegressor.
        """
        self.n_features = len(X[0]) if X else 0

        # Определяем, какие признаки будем использовать для построения дерева
        if feature_indices is None:
            n_feat = self._get_max_features(self.n_features)
            self.feature_indices = self.rng.sample(range(self.n_features), n_feat)
        else:
            self.feature_indices = feature_indices

        # Запускаем рекурсивное построение дерева
        self.root = self._build_tree(X, y, feature_indices=self.feature_indices)
        return self

    def _predict_one(self, node: NodeReg, x: List[float]) -> float:
        """
        Предсказывает значение для одного образца, проходя по дереву.

        Начиная с корня, на каждом внутреннем узле сравниваем значение
        признака с порогом и спускаемся в соответствующее поддерево.

        Args:
            node: Текущий узел дерева.
            x: Вектор признаков одного образца.

        Returns:
            Предсказанное числовое значение из листа.
        """
        if node.is_leaf():
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._predict_one(node.left, x)
        return self._predict_one(node.right, x)

    def predict(self, X: List[List[float]]) -> List[float]:
        """
        Предсказывает значения для набора образцов.

        Args:
            X: Матрица признаков (n_samples x n_features).

        Returns:
            Список предсказанных значений (длина = len(X)).
        """
        return [self._predict_one(self.root, x) for x in X]


class CustomRandomForest:
    """
    Случайный лес (Random Forest) для задачи регрессии.

    Ансамбль из n_estimators деревьев решений. Каждое дерево обучается
    на бутстрап-выборке (случайная выборка с повторениями из исходных данных),
    что снижает дисперсию и повышает обобщающую способность.

    Предсказание ансамбля — среднее арифметическое предсказаний всех деревьев.

    Attributes:
        n_estimators: Количество деревьев в лесу.
        max_depth: Максимальная глубина каждого дерева.
        min_samples_split: Минимальное число образцов для разбиения.
        max_features: Количество признаков для случайного подмножества.
        bootstrap: Использовать ли бутстрап-выборку для каждого дерева.
        random_state: Базовый seed для воспроизводимости.
        rng: Генератор случайных чисел.
        trees: Список обученных деревьев (DecisionTreeRegressor).
    """

    def __init__(self, n_estimators: int = 10, max_depth: int = 5,
                 min_samples_split: int = 2, max_features: str = 'sqrt',
                 bootstrap: bool = True, random_state: Optional[int] = None):
        """
        Инициализирует случайный лес.

        Args:
            n_estimators: Количество деревьев. По умолчанию 10.
            max_depth: Максимальная глубина каждого дерева. По умолчанию 5.
            min_samples_split: Минимальное число образцов для разбиения. По умолчанию 2.
            max_features: Количество признаков для подмножества.
                          По умолчанию 'sqrt' (квадратный корень от числа признаков).
            bootstrap: Если True, каждое дерево обучается на бутстрап-выборке.
                       Если False, на полном датасете.
            random_state: Seed для воспроизводимости.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.rng = random.Random(random_state)
        self.trees: List[DecisionTreeRegressor] = []

    def _bootstrap_sample(self, X: List[List[float]], y: List[float]) -> Tuple[List[List[float]], List[float]]:
        """
        Создаёт бутстрап-выборку (выборку с возвращением).

        Размер выборки равен исходному датасету. Некоторые образцы
        могут встречаться несколько раз, а некоторые — ни разу
        (это так называемые out-of-bag образцы).

        Args:
            X: Исходная матрица признаков.
            y: Исходный вектор целевых значений.

        Returns:
            Кортеж (X_boot, y_boot) — бутстрап-выборка признаков и целей.
        """
        n = len(X)
        # Генерируем n случайных индексов с повторениями из диапазона [0, n-1]
        indices = [self.rng.randint(0, n - 1) for _ in range(n)]
        X_boot = [X[i] for i in indices]
        y_boot = [y[i] for i in indices]
        return X_boot, y_boot

    def fit(self, X: List[List[float]], y: List[float]) -> 'CustomRandomForest':
        """
        Обучает случайный лес.

        Для каждого дерева:
            1. Создаётся новый DecisionTreeRegressor с уникальным random_state.
            2. Если bootstrap=True, генерируется бутстрап-выборка.
            3. Дерево обучается на соответствующих данных.
            4. Дерево добавляется в ансамбль.

        Args:
            X: Матрица признаков (n_samples x n_features).
            y: Вектор целевых значений (n_samples,).

        Returns:
            self — обученный экземпляр CustomRandomForest.
        """
        for i in range(self.n_estimators):
            # Создаём дерево с уникальным seed для воспроизводимости
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                random_state=self.random_state + i if self.random_state else None
            )

            # Выбираем данные для обучения текущего дерева
            if self.bootstrap:
                X_train, y_train = self._bootstrap_sample(X, y)
                tree.fit(X_train, y_train)
            else:
                tree.fit(X, y)

            self.trees.append(tree)

        return self

    def predict(self, X: List[List[float]]) -> List[float]:
        """
        Предсказывает значения с помощью ансамбля деревьев.

        Для каждого образца собираются предсказания всех деревьев,
        затем вычисляется среднее арифметическое.

        Args:
            X: Матрица признаков (n_samples x n_features).

        Returns:
            Список усреднённых предсказаний (длина = len(X)).
        """
        predictions = []
        for x in X:
            # Собираем предсказания всех деревьев для одного образца
            preds = [tree._predict_one(tree.root, x) for tree in self.trees]
            # Усредняем предсказания ансамбля
            predictions.append(sum(preds) / len(preds))
        return predictions