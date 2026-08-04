import csv
import random
import math
from collections import Counter
from typing import List

# sklearn для сравнения
from sklearn.ensemble import RandomForestRegressor as SklearnRF
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

from forest import CustomRandomForest

# ==========================================
# ЗАГРУЗКА ДАННЫХ
# ==========================================
def load_data(csv_path):
    X = []
    y = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            features = [
                int(row['seller_type']),
                int(row['object_type']),
                float(row['total_area']),
                float(row['kitchen_area']),
                int(row['rooms']),
                int(row['layout']),
                int(row['furnished']),
                int(row['renovation']),
            ]
            X.append(features)
            y.append(int(row['\ufeffprice']))
    return X, y


def train_test_split(X, y, test_size=0.2, random_state=None):
    rng = random.Random(random_state)
    indices = list(range(len(X)))
    rng.shuffle(indices)
    split = int(len(X) * (1 - test_size))
    train_idx = indices[:split]
    test_idx = indices[split:]
    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_test = [X[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]
    return X_train, X_test, y_train, y_test


def evaluate(y_true, y_pred, name):
    mae = sum(abs(a - b) for a, b in zip(y_true, y_pred)) / len(y_true)
    rmse = math.sqrt(sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / len(y_true))
    ss_res = sum((a - b) ** 2 for a, b in zip(y_true, y_pred))
    ss_tot = sum((yi - sum(y_true)/len(y_true)) ** 2 for yi in y_true)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    print(f"\n{'='*50}")
    print(f"{name}")
    print(f"{'='*50}")
    print(f"MAE:  {mae:>15,.0f} грн")
    print(f"RMSE: {rmse:>15,.0f} грн")
    print(f"R²:   {r2:>15.3f}")
    return mae, rmse, r2


# ==========================================
# ОСНОВНОЙ СКРИПТ
# ==========================================
if __name__ == "__main__":
    CSV_PATH = "apartments_dataset.csv"
    
    print("[*] Загрузка данных...")
    X, y = load_data(CSV_PATH)
    print(f"[✓] Загружено: {len(X)} записей, {len(X[0])} признаков")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"[*] Train: {len(X_train)}, Test: {len(X_test)}")
    
    # ==========================================
    # 1. КАСТОМНЫЙ СЛУЧАЙНЫЙ ЛЕС
    # ==========================================
    print("\n[*] Обучение КАСТОМНОГО случайного леса...")
    custom_rf = CustomRandomForest(
        n_estimators=100,
        max_depth=15,
        min_samples_split=3,
        max_features='sqrt',
        bootstrap=True,
        random_state=42
    )
    custom_rf.fit(X_train, y_train)
    custom_preds = custom_rf.predict(X_test)
    custom_mae, custom_rmse, custom_r2 = evaluate(y_test, custom_preds, "КАСТОМНЫЙ Random Forest")
    
    # ==========================================
    # 2. SKLEARN СЛУЧАЙНЫЙ ЛЕС
    # ==========================================
    print("\n[*] Обучение SKLEARN случайного леса...")
    sklearn_rf = SklearnRF(
        n_estimators=100,
        max_depth=15,
        min_samples_split=3,
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )
    sklearn_rf.fit(np.array(X_train), np.array(y_train))
    sklearn_preds = sklearn_rf.predict(np.array(X_test))
    sklearn_mae, sklearn_rmse, sklearn_r2 = evaluate(y_test, sklearn_preds, "SKLEARN Random Forest")
    
    # ==========================================
    # 3. СРАВНЕНИЕ
    # ==========================================
    print(f"\n{'='*50}")
    print("СРАВНЕНИЕ")
    print(f"{'='*50}")
    print(f"{'Метрика':<15} {'Кастомный':>15} {'Sklearn':>15} {'Разница':>15}")
    print(f"{'-'*60}")
    print(f"{'MAE':<15} {custom_mae:>15,.0f} {sklearn_mae:>15,.0f} {custom_mae-sklearn_mae:>15,.0f}")
    print(f"{'RMSE':<15} {custom_rmse:>15,.0f} {sklearn_rmse:>15,.0f} {custom_rmse-sklearn_rmse:>15,.0f}")
    print(f"{'R²':<15} {custom_r2:>15.3f} {sklearn_r2:>15.3f} {custom_r2-sklearn_r2:>15.3f}")
    
    # Примеры предсказаний
    print(f"\n{'='*50}")
    print("ПРИМЕРЫ ПРЕДСКАЗАНИЙ (первые 5)")
    print(f"{'='*50}")
    print(f"{'Реальная':>12} | {'Кастомный':>12} | {'Sklearn':>12} | {'Разн. каст':>10} | {'Разн. skl':>10}")
    print(f"{'-'*65}")
    for i in range(min(5, len(y_test))):
        print(f"{y_test[i]:>12,} | {custom_preds[i]:>12,.0f} | {sklearn_preds[i]:>12,.0f} | "
              f"{abs(y_test[i]-custom_preds[i]):>10,.0f} | {abs(y_test[i]-sklearn_preds[i]):>10,.0f}")
    
    # Важность признаков
    feature_names = ['seller_type', 'object_type', 
                     'total_area', 'kitchen_area', 'rooms', 'layout', 
                     'furnished', 'renovation']
    
    print(f"\n{'='*50}")
    print("ВАЖНОСТЬ ПРИЗНАКОВ (sklearn)")
    print(f"{'='*50}")
    for name, imp in zip(feature_names, sklearn_rf.feature_importances_):
        print(f"  {name:20s}: {imp:.3f}")