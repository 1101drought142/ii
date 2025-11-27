"""
Лабораторная работа 3: Классификация и нейронные сети
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, classification_report, 
                            confusion_matrix, roc_curve, auc)
from sklearn.multiclass import OneVsRestClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import config

# Создание папок для графиков и логов
os.makedirs('laba3', exist_ok=True)
os.makedirs(config.TENSORBOARD_DIR, exist_ok=True)

# Настройка визуализации
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = config.FIG_SIZE
plt.rcParams['figure.dpi'] = config.DPI

# Фиксация seed для воспроизводимости
np.random.seed(config.RANDOM_STATE)
tf.random.set_seed(config.RANDOM_STATE)

def load_data():
    """Загрузка и анализ данных"""
    df = pd.read_csv(config.DATA_PATH)
    print("Форма данных:", df.shape)
    print("\nПервые строки:")
    print(df.head())
    print("\nИнформация о данных:")
    print(df.info())
    print("\nРаспределение классов:")
    print(df['Species'].value_counts())
    return df

def preprocess_data(df):
    """Предобработка данных"""
    # Удаление ID если есть
    if 'Id' in df.columns:
        df = df.drop('Id', axis=1)
    
    # Разделение на признаки и целевую переменную
    X = df.drop('Species', axis=1)
    y = df['Species']
    
    # Кодирование меток
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Стандартизация
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"\nПризнаки: {list(X.columns)}")
    print(f"Классы: {le.classes_}")
    
    return X_scaled, y_encoded, le, scaler

def train_classifiers(X_train, X_test, y_train, y_test):
    """Обучение классификаторов"""
    classifiers = {
        'GaussianNB': GaussianNB(),
        'MultinomialNB': MultinomialNB(),
        'ComplementNB': ComplementNB(),
        'BernoulliNB': BernoulliNB(),
        'Decision Tree': DecisionTreeClassifier(random_state=config.RANDOM_STATE),
        'LDA': LinearDiscriminantAnalysis(),
        'SVM': SVC(random_state=config.RANDOM_STATE, probability=True),
        'KNN': KNeighborsClassifier()
    }
    
    results = {}
    
    for name, clf in classifiers.items():
        print(f"\nОбучение {name}...")
        
        # Обучение
        clf.fit(X_train, y_train)
        
        # Предсказания
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test) if hasattr(clf, 'predict_proba') else None
        
        # Метрики
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # ROC-AUC (для многоклассовой задачи)
        if y_pred_proba is not None:
            try:
                roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
            except:
                roc_auc = 0.0
        else:
            roc_auc = 0.0
        
        # Кросс-валидация
        cv_scores = cross_val_score(clf, X_train, y_train, cv=config.CV_FOLDS, scoring='accuracy')
        
        results[name] = {
            'classifier': clf,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  ROC-AUC: {roc_auc:.4f}")
        print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    return results

def tune_hyperparameters(X_train, y_train):
    """Настройка гиперпараметров"""
    print("\n" + "="*60)
    print("НАСТРОЙКА ГИПЕРПАРАМЕТРОВ")
    print("="*60)
    
    tuned_results = {}
    
    # Decision Tree
    print("\nНастройка Decision Tree...")
    param_grid_dt = {
        'max_depth': [3, 5, 7, 10, None],
        'min_samples_split': [2, 5, 10]
    }
    dt = DecisionTreeClassifier(random_state=config.RANDOM_STATE)
    grid_dt = GridSearchCV(dt, param_grid_dt, cv=config.CV_FOLDS, scoring='accuracy')
    grid_dt.fit(X_train, y_train)
    print(f"Лучшие параметры: {grid_dt.best_params_}")
    print(f"Лучший score: {grid_dt.best_score_:.4f}")
    tuned_results['Decision Tree'] = grid_dt.best_estimator_
    
    # SVM
    print("\nНастройка SVM...")
    param_grid_svm = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto']
    }
    svm = SVC(random_state=config.RANDOM_STATE, probability=True)
    grid_svm = GridSearchCV(svm, param_grid_svm, cv=config.CV_FOLDS, scoring='accuracy')
    grid_svm.fit(X_train, y_train)
    print(f"Лучшие параметры: {grid_svm.best_params_}")
    print(f"Лучший score: {grid_svm.best_score_:.4f}")
    tuned_results['SVM'] = grid_svm.best_estimator_
    
    # KNN
    print("\nНастройка KNN...")
    param_grid_knn = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    knn = KNeighborsClassifier()
    grid_knn = GridSearchCV(knn, param_grid_knn, cv=config.CV_FOLDS, scoring='accuracy')
    grid_knn.fit(X_train, y_train)
    print(f"Лучшие параметры: {grid_knn.best_params_}")
    print(f"Лучший score: {grid_knn.best_score_:.4f}")
    tuned_results['KNN'] = grid_knn.best_estimator_
    
    return tuned_results

def build_neural_network(input_dim, num_classes, hidden_layers=[64, 32], learning_rate=0.001):
    """Построение нейронной сети"""
    model = keras.Sequential()
    
    # Входной слой
    model.add(layers.Dense(hidden_layers[0], activation='relu', input_dim=input_dim))
    model.add(layers.Dropout(0.2))
    
    # Скрытые слои
    for units in hidden_layers[1:]:
        model.add(layers.Dense(units, activation='relu'))
        model.add(layers.Dropout(0.2))
    
    # Выходной слой
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    # Компиляция
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

def train_neural_network(X_train, X_test, y_train, y_test, num_classes):
    """Обучение нейронной сети"""
    print("\n" + "="*60)
    print("ОБУЧЕНИЕ НЕЙРОННОЙ СЕТИ")
    print("="*60)
    
    # Создание модели
    model = build_neural_network(X_train.shape[1], num_classes)
    print("\nАрхитектура модели:")
    model.summary()
    
    # Callbacks
    callbacks = [
        keras.callbacks.TensorBoard(log_dir=config.TENSORBOARD_DIR, histogram_freq=1),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    ]
    
    # Обучение
    history = model.fit(X_train, y_train,
                       epochs=config.EPOCHS,
                       batch_size=config.BATCH_SIZE,
                       validation_split=config.VALIDATION_SPLIT,
                       callbacks=callbacks,
                       verbose=1)
    
    # Оценка
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Предсказания
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Метрики
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    try:
        roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
    except:
        roc_auc = 0.0
    
    # Визуализация обучения
    plot_training_history(history)
    
    return {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'history': history,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

def plot_training_history(history):
    """Визуализация истории обучения"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Model Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('laba3/neural_network_training.png', dpi=config.DPI, bbox_inches='tight')
    plt.close()

def compare_results(results, nn_results=None):
    """Сравнение результатов всех методов"""
    comparison = pd.DataFrame({
        'Метод': [],
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1-Score': [],
        'ROC-AUC': []
    })
    
    for name, res in results.items():
        comparison = pd.concat([comparison, pd.DataFrame({
            'Метод': [name],
            'Accuracy': [res['accuracy']],
            'Precision': [res['precision']],
            'Recall': [res['recall']],
            'F1-Score': [res['f1']],
            'ROC-AUC': [res['roc_auc']]
        })], ignore_index=True)
    
    if nn_results:
        comparison = pd.concat([comparison, pd.DataFrame({
            'Метод': ['Neural Network'],
            'Accuracy': [nn_results['accuracy']],
            'Precision': [nn_results['precision']],
            'Recall': [nn_results['recall']],
            'F1-Score': [nn_results['f1']],
            'ROC-AUC': [nn_results['roc_auc']]
        })], ignore_index=True)
    
    print("\n" + "="*60)
    print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
    print("="*60)
    print(comparison.to_string(index=False))
    
    # Визуализация
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    for i, metric in enumerate(metrics):
        axes[i].barh(comparison['Метод'], comparison[metric], color='steelblue')
        axes[i].set_xlabel(metric)
        axes[i].set_title(f'Сравнение {metric}')
        axes[i].grid(True, alpha=0.3)
    
    # Общая визуализация
    comparison_melted = comparison.melt(id_vars='Метод', value_vars=metrics, 
                                        var_name='Метрика', value_name='Значение')
    sns.barplot(data=comparison_melted, x='Метод', y='Значение', hue='Метрика', ax=axes[5])
    axes[5].set_title('Все метрики')
    axes[5].tick_params(axis='x', rotation=45)
    axes[5].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('laba3/comparison.png', dpi=config.DPI, bbox_inches='tight')
    plt.close()
    
    return comparison

def visualize_confusion_matrices(results, y_test, le):
    """Визуализация матриц ошибок"""
    n_classifiers = len(results)
    fig, axes = plt.subplots(2, (n_classifiers + 1) // 2, figsize=(16, 10))
    axes = axes.flatten()
    
    for idx, (name, res) in enumerate(results.items()):
        cm = confusion_matrix(y_test, res['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   xticklabels=le.classes_, yticklabels=le.classes_)
        axes[idx].set_title(f'{name}')
        axes[idx].set_ylabel('Истинный класс')
        axes[idx].set_xlabel('Предсказанный класс')
    
    plt.tight_layout()
    plt.savefig('laba3/confusion_matrices.png', dpi=config.DPI, bbox_inches='tight')
    plt.close()

def main():
    """Основная функция"""
    print("="*60)
    print("ЛАБОРАТОРНАЯ РАБОТА 3: КЛАССИФИКАЦИЯ И НЕЙРОННЫЕ СЕТИ")
    print("="*60)
    
    # Загрузка данных
    df = load_data()
    
    # Предобработка
    X, y, le, scaler = preprocess_data(df)
    
    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )
    
    print(f"\nРазмер обучающей выборки: {X_train.shape[0]}")
    print(f"Размер тестовой выборки: {X_test.shape[0]}")
    
    # Обучение классификаторов
    print("\n" + "="*60)
    print("ОБУЧЕНИЕ КЛАССИФИКАТОРОВ")
    print("="*60)
    results = train_classifiers(X_train, X_test, y_train, y_test)
    
    # Настройка гиперпараметров
    tuned_classifiers = tune_hyperparameters(X_train, y_train)
    
    # Оценка настроенных классификаторов
    print("\n" + "="*60)
    print("ОЦЕНКА НАСТРОЕННЫХ КЛАССИФИКАТОРОВ")
    print("="*60)
    for name, clf in tuned_classifiers.items():
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{name}: Accuracy = {accuracy:.4f}")
        if name in results:
            results[name]['accuracy'] = accuracy
            results[name]['y_pred'] = y_pred
    
    # Обучение нейронной сети
    num_classes = len(le.classes_)
    nn_results = train_neural_network(X_train, X_test, y_train, y_test, num_classes)
    
    # Сравнение результатов
    comparison = compare_results(results, nn_results)
    
    # Визуализация матриц ошибок
    visualize_confusion_matrices(results, y_test, le)
    
    print("\n" + "="*60)
    print("РАБОТА ЗАВЕРШЕНА")
    print("="*60)
    print(f"\nTensorBoard логи сохранены в: {config.TENSORBOARD_DIR}")
    print("Запустите: tensorboard --logdir=" + config.TENSORBOARD_DIR)

if __name__ == '__main__':
    main()

