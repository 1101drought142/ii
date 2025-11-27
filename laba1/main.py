"""
Лабораторная работа 1: Линейная регрессия и факторный анализ
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
import config

# Создание папки для графиков
os.makedirs('laba1', exist_ok=True)
# Создаем папку photos в корне проекта
photos_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'photos')
os.makedirs(photos_dir, exist_ok=True)

# Настройка визуализации
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = config.FIG_SIZE
plt.rcParams['figure.dpi'] = config.DPI

def load_data():
    """Загрузка и первичный анализ данных"""
    df = pd.read_csv(config.DATA_PATH)
    print("Форма данных:", df.shape)
    print("\nПервые строки:")
    print(df.head())
    print("\nИнформация о данных:")
    print(df.info())
    print("\nОписательная статистика:")
    print(df.describe())
    return df

def preprocess_data(df):
    """Предобработка данных"""
    # Удаление пропусков
    df = df.dropna()
    
    # Разделение на признаки и целевую переменную
    y = df['price'].copy()
    X = df.drop('price', axis=1).copy()
    
    print(f"\nИсходные признаки: {list(X.columns)}")
    print(f"Целевая переменная: price")
    print(f"Исходный размер данных: {len(X)}")
    
    # Определение типов признаков
    numeric_cols = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
    binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
    categorical_cols = ['furnishingstatus']
    
    # Кодирование бинарных признаков (yes/no -> 1/0)
    for col in binary_cols:
        if col in X.columns:
            X[col] = X[col].map({'yes': 1, 'no': 0})
    
    # One-hot encoding для категориальной переменной
    if 'furnishingstatus' in X.columns:
        furnishing_dummies = pd.get_dummies(X['furnishingstatus'], prefix='furnishing')
        X = pd.concat([X.drop('furnishingstatus', axis=1), furnishing_dummies], axis=1)
    
    print(f"\nПризнаки после кодирования: {list(X.columns)}")
    print(f"Количество признаков: {X.shape[1]}")
    
    # Проверка типов данных
    print("\nТипы данных:")
    print(X.dtypes)
    
    print("\nОписательная статистика исходных данных:")
    print(X.describe())
    
    return X, y

def visualize_data(X, y):
    """Визуализация распределения признаков и целевой переменной"""
    # Выбираем важные признаки для визуализации
    # Основные числовые признаки (непрерывные)
    main_numeric = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
    main_numeric_cols = [col for col in main_numeric if col in X.columns]
    
    # Категориальные признаки (one-hot encoded)
    furnishing_cols = [col for col in X.columns if col.startswith('furnishing_')]
    
    # Объединяем все признаки для визуализации
    all_cols = main_numeric_cols + furnishing_cols
    
    # Добавляем целевую переменную в конец
    total_plots = len(all_cols) + 1
    
    n_cols = 4
    n_rows = (total_plots + n_cols - 1) // n_cols  # Округление вверх
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1) if hasattr(axes, 'reshape') else [axes]
    axes = axes.flatten()
    
    # Распределение признаков
    for i, col in enumerate(all_cols):
        if i < len(axes):
            if col in furnishing_cols:
                # Для one-hot признаков используем bar plot
                value_counts = X[col].value_counts().sort_index()
                axes[i].bar(value_counts.index.astype(str), value_counts.values, 
                           edgecolor='black', alpha=0.7, color='steelblue')
            else:
                # Для непрерывных признаков используем histogram
                axes[i].hist(X[col], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
            
            # Улучшенные подписи для категориальных признаков
            title = col.replace('_', ' ').title() if '_' in col else col.title()
            axes[i].set_title(f'Распределение {title}', fontsize=10)
            axes[i].set_xlabel(title, fontsize=8)
            axes[i].set_ylabel('Частота', fontsize=8)
            axes[i].tick_params(labelsize=8)
            axes[i].grid(True, alpha=0.3, axis='y')
    
    # Распределение целевой переменной (в последней позиции)
    target_idx = len(all_cols)
    if target_idx < len(axes):
        axes[target_idx].hist(y, bins=30, edgecolor='black', color='green', alpha=0.7)
        axes[target_idx].set_title('Распределение целевой переменной\n(Price)', fontsize=10)
        axes[target_idx].set_xlabel('Цена (Price)', fontsize=8)
        axes[target_idx].set_ylabel('Частота', fontsize=8)
        axes[target_idx].tick_params(labelsize=8)
        axes[target_idx].grid(True, alpha=0.3, axis='y')
    
    # Скрыть лишние subplot'ы
    for i in range(target_idx + 1, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    photos_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'photos', 'distributions.png')
    plt.savefig(photos_path, dpi=config.DPI, bbox_inches='tight')
    plt.close()

def correlation_analysis(X, y):
    """Анализ корреляций и мультиколлинеарности"""
    # Добавляем целевую переменную для анализа корреляций
    X_with_target = X.copy()
    X_with_target['price'] = y
    
    corr_matrix = X_with_target.corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Матрица корреляций', fontsize=14, pad=20)
    plt.tight_layout()
    photos_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'photos', 'correlation_matrix.png')
    plt.savefig(photos_path, dpi=config.DPI, bbox_inches='tight')
    plt.close()
    
    # Расчет VIF (только для признаков, без целевой переменной)
    # Удаляем один из one-hot encoded признаков для избежания мультиколлинеарности
    X_for_vif = X.copy()
    furnishing_cols = [col for col in X_for_vif.columns if col.startswith('furnishing_')]
    if len(furnishing_cols) > 0:
        # Удаляем последний one-hot признак (остальные линейно зависимы от него)
        X_for_vif = X_for_vif.drop(furnishing_cols[-1], axis=1)
    
    # ВАЖНО: VIF рассчитывается на исходных данных БЕЗ нормализации
    # VIF требует числовые данные, но не требует их нормализации
    # Убеждаемся, что все данные числовые
    X_for_vif = X_for_vif.select_dtypes(include=[np.number])
    # Заменяем бесконечные значения на NaN, затем удаляем их
    X_for_vif = X_for_vif.replace([np.inf, -np.inf], np.nan)
    X_for_vif = X_for_vif.fillna(X_for_vif.mean())
    
    # Преобразуем в numpy array для VIF
    X_vif_array = X_for_vif.values.astype(float)
    
    vif_data = pd.DataFrame()
    vif_data["Признак"] = X_for_vif.columns
    vif_data["VIF"] = [variance_inflation_factor(X_vif_array, i) for i in range(X_vif_array.shape[1])]
    vif_data = vif_data.sort_values('VIF', ascending=False)
    
    print("\nVIF коэффициенты:")
    print(vif_data.to_string(index=False))
    
    # Вывод признаков с высокой мультиколлинеарностью
    high_vif = vif_data[vif_data['VIF'] > 10]
    if len(high_vif) > 0:
        print(f"\nПризнаки с высокой мультиколлинеарностью (VIF > 10):")
        print(high_vif.to_string(index=False))
    else:
        print("\nМультиколлинеарность не обнаружена (все VIF < 10)")
    
    return corr_matrix, vif_data

def calculate_metrics(y_true, y_pred):
    """Расчет метрик качества"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    
    return {'RMSE': rmse, 'R2': r2, 'MAPE': mape}

def train_models(X_train, X_test, y_train, y_test, scaler=None):
    """Обучение моделей линейной и гребневой регрессии"""
    if scaler:
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test
    
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=config.RIDGE_ALPHA)
    }
    
    results = {}
    
    for name, model in models.items():
        # Обучение
        model.fit(X_train_scaled, y_train)
        
        # Предсказания
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        
        # Метрики
        train_metrics = calculate_metrics(y_train, y_train_pred)
        test_metrics = calculate_metrics(y_test, y_test_pred)
        
        # Кросс-валидация
        cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                                   cv=config.CV_FOLDS, scoring='r2')
        
        results[name] = {
            'model': model,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        print(f"\n{name}:")
        print(f"  Train - RMSE: {train_metrics['RMSE']:.4f}, R2: {train_metrics['R2']:.4f}, MAPE: {train_metrics['MAPE']:.2f}%")
        print(f"  Test  - RMSE: {test_metrics['RMSE']:.4f}, R2: {test_metrics['R2']:.4f}, MAPE: {test_metrics['MAPE']:.2f}%")
        print(f"  CV R2: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    return results

def apply_pca(X_train, X_test):
    """Применение PCA для снижения размерности"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Определение количества компонент
    pca = PCA()
    pca.fit(X_train_scaled)
    
    cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumsum_variance >= config.PCA_VARIANCE_THRESHOLD) + 1
    
    print(f"\nКоличество компонент для {config.PCA_VARIANCE_THRESHOLD*100}% дисперсии: {n_components}")
    
    # График каменистой осыпи
    # Два subplot'а: один для накопленной дисперсии, другой для индивидуальной
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Накопленная объясненная дисперсия
    ax1.plot(range(1, len(cumsum_variance) + 1), cumsum_variance, 'bo-', linewidth=2, markersize=8)
    ax1.axhline(y=config.PCA_VARIANCE_THRESHOLD, color='r', linestyle='--', linewidth=2,
                label=f'{config.PCA_VARIANCE_THRESHOLD*100}% дисперсии')
    ax1.axvline(x=n_components, color='g', linestyle='--', linewidth=2,
                label=f'Выбрано компонент: {n_components}')
    ax1.set_xlabel('Количество компонент', fontsize=12)
    ax1.set_ylabel('Накопленная объясненная дисперсия', fontsize=12)
    ax1.set_title('Накопленная объясненная дисперсия (PCA)', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Индивидуальная объясненная дисперсия
    individual_variance = pca.explained_variance_ratio_
    ax2.bar(range(1, len(individual_variance) + 1), individual_variance, alpha=0.7, color='steelblue')
    ax2.set_xlabel('Номер компоненты', fontsize=12)
    ax2.set_ylabel('Объясненная дисперсия', fontsize=12)
    ax2.set_title('Индивидуальная объясненная дисперсия', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    photos_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'photos', 'pca_scree.png')
    plt.savefig(photos_path, dpi=config.DPI, bbox_inches='tight')
    plt.close()
    
    # Применение PCA
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    print(f"Объясненная дисперсия: {pca.explained_variance_ratio_.sum():.4f}")
    
    return X_train_pca, X_test_pca, scaler

def compare_results(results_original, results_pca):
    """Сравнение результатов моделей"""
    comparison = pd.DataFrame({
        'Модель': [],
        'Данные': [],
        'RMSE': [],
        'R2': [],
        'MAPE': []
    })
    
    for name in results_original.keys():
        comparison = pd.concat([comparison, pd.DataFrame({
            'Модель': [name],
            'Данные': ['Исходные'],
            'RMSE': [results_original[name]['test_metrics']['RMSE']],
            'R2': [results_original[name]['test_metrics']['R2']],
            'MAPE': [results_original[name]['test_metrics']['MAPE']]
        })], ignore_index=True)
        
        comparison = pd.concat([comparison, pd.DataFrame({
            'Модель': [name],
            'Данные': ['PCA'],
            'RMSE': [results_pca[name]['test_metrics']['RMSE']],
            'R2': [results_pca[name]['test_metrics']['R2']],
            'MAPE': [results_pca[name]['test_metrics']['MAPE']]
        })], ignore_index=True)
    
    print("\n" + "="*60)
    print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
    print("="*60)
    print(comparison.to_string(index=False))
    
    return comparison

def main():
    """Основная функция"""
    print("="*60)
    print("ЛАБОРАТОРНАЯ РАБОТА 1: ЛИНЕЙНАЯ РЕГРЕССИЯ")
    print("="*60)
    
    # Загрузка данных
    df = load_data()
    
    # Предобработка
    X, y = preprocess_data(df)
    
    # Визуализация
    visualize_data(X, y)
    
    # Анализ корреляций
    corr_matrix, vif_data = correlation_analysis(X, y)
    
    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    
    print(f"\nРазмер обучающей выборки: {X_train.shape[0]}")
    print(f"Размер тестовой выборки: {X_test.shape[0]}")
    
    # Обучение моделей на исходных данных
    print("\n" + "="*60)
    print("МОДЕЛИ НА ИСХОДНЫХ ДАННЫХ")
    print("="*60)
    scaler_original = StandardScaler()
    results_original = train_models(X_train, X_test, y_train, y_test, scaler_original)
    
    # Применение PCA
    print("\n" + "="*60)
    print("ПРИМЕНЕНИЕ PCA")
    print("="*60)
    X_train_pca, X_test_pca, scaler_pca = apply_pca(X_train, X_test)
    
    # Обучение моделей на PCA компонентах
    print("\n" + "="*60)
    print("МОДЕЛИ НА PCA КОМПОНЕНТАХ")
    print("="*60)
    results_pca = train_models(X_train_pca, X_test_pca, y_train, y_test)
    
    # Сравнение результатов
    comparison = compare_results(results_original, results_pca)
    
    print("\n" + "="*60)
    print("РАБОТА ЗАВЕРШЕНА")
    print("="*60)

if __name__ == '__main__':
    main()

