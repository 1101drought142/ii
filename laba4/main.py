"""
Лабораторная работа 4: Кластеризация
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import (silhouette_score, calinski_harabasz_score, 
                            davies_bouldin_score, adjusted_rand_score,
                            normalized_mutual_info_score, homogeneity_score,
                            completeness_score, v_measure_score)
from sklearn.model_selection import train_test_split
from scipy.cluster.hierarchy import dendrogram, linkage
import config

# Создание папки для графиков
os.makedirs('laba4', exist_ok=True)

# Настройка визуализации
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = config.FIG_SIZE
plt.rcParams['figure.dpi'] = config.DPI

def load_data():
    """Загрузка и анализ данных"""
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
    # Удаление ID если есть
    if 'Id' in df.columns:
        df = df.drop('Id', axis=1)
    
    # Сохранение меток классов если есть (для внешних метрик)
    y_true = None
    if 'Species' in df.columns:
        y_true = df['Species'].copy()
        df = df.drop('Species', axis=1)
    
    # Стандартизация
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    
    print(f"\nПризнаки: {list(df.columns)}")
    print(f"Размерность после стандартизации: {X_scaled.shape}")
    
    return X_scaled, df.columns, y_true, scaler

def visualize_data(X, feature_names, y_true=None):
    """Визуализация данных"""
    # Матрица диаграмм рассеивания
    df_viz = pd.DataFrame(X, columns=feature_names)
    if y_true is not None:
        df_viz['Species'] = y_true
    
    if y_true is not None:
        sns.pairplot(df_viz, hue='Species', diag_kind='hist')
    else:
        sns.pairplot(df_viz, diag_kind='hist')
    
    plt.suptitle('Матрица диаграмм рассеивания', y=1.02)
    plt.tight_layout()
    plt.savefig('laba4/pairplot.png', dpi=config.DPI, bbox_inches='tight')
    plt.close()
    
    # Гистограммы распределения
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(feature_names):
        axes[i].hist(X[:, i], bins=30, edgecolor='black', alpha=0.7)
        axes[i].set_title(f'Распределение {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Частота')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('laba4/distributions.png', dpi=config.DPI, bbox_inches='tight')
    plt.close()

def find_optimal_k_elbow(X):
    """Метод локтя для определения оптимального k"""
    inertias = []
    k_range = list(config.K_RANGE)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, n_init=config.N_INIT, 
                       max_iter=config.MAX_ITER, random_state=config.RANDOM_STATE)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertias, 'bo-')
    plt.xlabel('Количество кластеров (k)')
    plt.ylabel('Инерция (Inertia)')
    plt.title('Метод локтя для определения оптимального k')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('laba4/elbow_method.png', dpi=config.DPI, bbox_inches='tight')
    plt.close()
    
    return inertias

def find_optimal_k_silhouette(X):
    """Силуэтный анализ для определения оптимального k"""
    silhouette_scores = []
    k_range = list(config.K_RANGE)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, n_init=config.N_INIT,
                       max_iter=config.MAX_ITER, random_state=config.RANDOM_STATE)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        silhouette_scores.append(score)
    
    optimal_k = k_range[np.argmax(silhouette_scores)]
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, silhouette_scores, 'ro-')
    plt.axvline(x=optimal_k, color='green', linestyle='--', 
               label=f'Оптимальное k = {optimal_k}')
    plt.xlabel('Количество кластеров (k)')
    plt.ylabel('Силуэтный коэффициент')
    plt.title('Силуэтный анализ для определения оптимального k')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('laba4/silhouette_analysis.png', dpi=config.DPI, bbox_inches='tight')
    plt.close()
    
    print(f"\nОптимальное k по силуэтному анализу: {optimal_k}")
    return silhouette_scores, optimal_k

def kmeans_clustering(X, n_clusters):
    """Кластеризация методом K-means"""
    kmeans = KMeans(n_clusters=n_clusters, n_init=config.N_INIT,
                   max_iter=config.MAX_ITER, random_state=config.RANDOM_STATE)
    labels = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_
    
    return labels, centers, kmeans

def hierarchical_clustering(X, n_clusters, linkage_method='ward'):
    """Иерархическая кластеризация"""
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
    labels = clustering.fit_predict(X)
    
    return labels, clustering

def plot_dendrogram(X, linkage_method='ward'):
    """Построение дендрограммы"""
    linkage_matrix = linkage(X, method=linkage_method)
    
    plt.figure(figsize=(12, 8))
    dendrogram(linkage_matrix, truncate_mode='level', p=10)
    plt.title(f'Дендрограмма (linkage: {linkage_method})')
    plt.xlabel('Образцы')
    plt.ylabel('Расстояние')
    plt.tight_layout()
    plt.savefig(f'laba4/dendrogram_{linkage_method}.png', dpi=config.DPI, bbox_inches='tight')
    plt.close()

def dbscan_clustering(X, eps, min_samples):
    """Кластеризация методом DBSCAN"""
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    return labels, dbscan, n_clusters, n_noise

def gaussian_mixture_clustering(X, n_components):
    """Кластеризация методом Gaussian Mixture"""
    gmm = GaussianMixture(n_components=n_components, random_state=config.RANDOM_STATE)
    labels = gmm.fit_predict(X)
    
    return labels, gmm

def calculate_metrics(X, labels, y_true=None):
    """Расчет метрик качества кластеризации"""
    metrics = {}
    
    # Внутренние метрики
    if len(set(labels)) > 1 and -1 not in labels:  # DBSCAN может иметь -1
        metrics['Silhouette Score'] = silhouette_score(X, labels)
        metrics['Calinski-Harabasz Index'] = calinski_harabasz_score(X, labels)
        metrics['Davies-Bouldin Index'] = davies_bouldin_score(X, labels)
    else:
        metrics['Silhouette Score'] = -1
        metrics['Calinski-Harabasz Index'] = 0
        metrics['Davies-Bouldin Index'] = float('inf')
    
    # Внешние метрики (если известны истинные классы)
    if y_true is not None:
        # Кодирование меток для сравнения
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_encoded = le.fit_transform(y_true)
        
        metrics['Adjusted Rand Index'] = adjusted_rand_score(y_encoded, labels)
        metrics['Normalized Mutual Info'] = normalized_mutual_info_score(y_encoded, labels)
        metrics['Homogeneity'] = homogeneity_score(y_encoded, labels)
        metrics['Completeness'] = completeness_score(y_encoded, labels)
        metrics['V-measure'] = v_measure_score(y_encoded, labels)
    
    return metrics

def visualize_clusters(X, labels, centers=None, method_name='', y_true=None):
    """Визуализация кластеров с использованием PCA"""
    pca = PCA(n_components=config.PCA_COMPONENTS)
    X_pca = pca.fit_transform(X)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Визуализация кластеров
    scatter = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, 
                              cmap='viridis', alpha=0.6, s=50)
    if centers is not None:
        centers_pca = pca.transform(centers)
        axes[0].scatter(centers_pca[:, 0], centers_pca[:, 1], 
                       c='red', marker='x', s=200, linewidths=3, label='Центры')
        axes[0].legend()
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    axes[0].set_title(f'Кластеры: {method_name}')
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[0])
    
    # Сравнение с истинными классами (если есть)
    if y_true is not None:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_encoded = le.fit_transform(y_true)
        scatter2 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y_encoded,
                                  cmap='Set1', alpha=0.6, s=50)
        axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        axes[1].set_title('Истинные классы')
        axes[1].grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=axes[1])
    else:
        axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'laba4/clusters_{method_name.lower().replace(" ", "_")}.png', 
               dpi=config.DPI, bbox_inches='tight')
    plt.close()

def analyze_cluster_centers(centers, feature_names):
    """Анализ центров кластеров"""
    centers_df = pd.DataFrame(centers, columns=feature_names)
    print("\nЦентры кластеров:")
    print(centers_df)
    
    # Визуализация
    plt.figure(figsize=(12, 6))
    centers_df.T.plot(kind='bar', figsize=(12, 6))
    plt.title('Центры кластеров по признакам')
    plt.xlabel('Признаки')
    plt.ylabel('Значения')
    plt.legend(title='Кластер')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('laba4/cluster_centers.png', dpi=config.DPI, bbox_inches='tight')
    plt.close()

def experiment_with_parameters(X, method='kmeans'):
    """Исследование влияния параметров на качество кластеризации"""
    print(f"\n" + "="*60)
    print(f"ЭКСПЕРИМЕНТЫ С ПАРАМЕТРАМИ: {method.upper()}")
    print("="*60)
    
    if method == 'kmeans':
        k_range = list(config.K_RANGE)
        silhouette_scores = []
        ch_scores = []
        
        for k in k_range:
            labels, _, _ = kmeans_clustering(X, k)
            if len(set(labels)) > 1:
                silhouette_scores.append(silhouette_score(X, labels))
                ch_scores.append(calinski_harabasz_score(X, labels))
            else:
                silhouette_scores.append(-1)
                ch_scores.append(0)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(k_range, silhouette_scores, 'o-')
        axes[0].set_xlabel('Количество кластеров (k)')
        axes[0].set_ylabel('Силуэтный коэффициент')
        axes[0].set_title('Влияние k на силуэтный коэффициент')
        axes[0].grid(True)
        
        axes[1].plot(k_range, ch_scores, 'o-', color='orange')
        axes[1].set_xlabel('Количество кластеров (k)')
        axes[1].set_ylabel('Calinski-Harabasz Index')
        axes[1].set_title('Влияние k на Calinski-Harabasz Index')
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig('laba4/parameter_experiments_kmeans.png', dpi=config.DPI, bbox_inches='tight')
        plt.close()

def main():
    """Основная функция"""
    print("="*60)
    print("ЛАБОРАТОРНАЯ РАБОТА 4: КЛАСТЕРИЗАЦИЯ")
    print("="*60)
    
    # Загрузка данных
    df = load_data()
    
    # Предобработка
    X, feature_names, y_true, scaler = preprocess_data(df)
    
    # Визуализация данных
    visualize_data(X, feature_names, y_true)
    
    # Подбор оптимального k
    print("\n" + "="*60)
    print("ПОДБОР ОПТИМАЛЬНОГО КОЛИЧЕСТВА КЛАСТЕРОВ")
    print("="*60)
    inertias = find_optimal_k_elbow(X)
    silhouette_scores, optimal_k = find_optimal_k_silhouette(X)
    
    # K-means кластеризация
    if config.USE_KMEANS:
        print("\n" + "="*60)
        print("КЛАСТЕРИЗАЦИЯ K-MEANS")
        print("="*60)
        labels_kmeans, centers, kmeans_model = kmeans_clustering(X, optimal_k)
        metrics_kmeans = calculate_metrics(X, labels_kmeans, y_true)
        
        print("\nМетрики K-means:")
        for metric, value in metrics_kmeans.items():
            print(f"  {metric}: {value:.4f}")
        
        visualize_clusters(X, labels_kmeans, centers, 'K-means', y_true)
        analyze_cluster_centers(centers, feature_names)
        experiment_with_parameters(X, 'kmeans')
    
    # Иерархическая кластеризация
    if config.USE_HIERARCHICAL:
        print("\n" + "="*60)
        print("ИЕРАРХИЧЕСКАЯ КЛАСТЕРИЗАЦИЯ")
        print("="*60)
        
        # Построение дендрограммы
        plot_dendrogram(X, 'ward')
        
        labels_hierarchical, hierarchical_model = hierarchical_clustering(X, optimal_k, 'ward')
        metrics_hierarchical = calculate_metrics(X, labels_hierarchical, y_true)
        
        print("\nМетрики Иерархическая кластеризация:")
        for metric, value in metrics_hierarchical.items():
            print(f"  {metric}: {value:.4f}")
        
        visualize_clusters(X, labels_hierarchical, None, 'Hierarchical', y_true)
    
    # DBSCAN (опционально)
    if config.USE_DBSCAN:
        print("\n" + "="*60)
        print("КЛАСТЕРИЗАЦИЯ DBSCAN")
        print("="*60)
        labels_dbscan, dbscan_model, n_clusters, n_noise = dbscan_clustering(X, eps=0.5, min_samples=5)
        print(f"Найдено кластеров: {n_clusters}, Шум: {n_noise}")
        
        if n_clusters > 0:
            metrics_dbscan = calculate_metrics(X, labels_dbscan, y_true)
            print("\nМетрики DBSCAN:")
            for metric, value in metrics_dbscan.items():
                print(f"  {metric}: {value:.4f}")
            visualize_clusters(X, labels_dbscan, None, 'DBSCAN', y_true)
    
    # Gaussian Mixture (опционально)
    if config.USE_GAUSSIAN_MIXTURE:
        print("\n" + "="*60)
        print("КЛАСТЕРИЗАЦИЯ GAUSSIAN MIXTURE")
        print("="*60)
        labels_gmm, gmm_model = gaussian_mixture_clustering(X, optimal_k)
        metrics_gmm = calculate_metrics(X, labels_gmm, y_true)
        
        print("\nМетрики Gaussian Mixture:")
        for metric, value in metrics_gmm.items():
            print(f"  {metric}: {value:.4f}")
        
        visualize_clusters(X, labels_gmm, None, 'Gaussian Mixture', y_true)
    
    # Сравнение результатов
    print("\n" + "="*60)
    print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
    print("="*60)
    
    comparison = pd.DataFrame()
    if config.USE_KMEANS:
        comparison = pd.concat([comparison, pd.DataFrame({
            'Метод': ['K-means'],
            **{k: [v] for k, v in metrics_kmeans.items()}
        })], ignore_index=True)
    
    if config.USE_HIERARCHICAL:
        comparison = pd.concat([comparison, pd.DataFrame({
            'Метод': ['Hierarchical'],
            **{k: [v] for k, v in metrics_hierarchical.items()}
        })], ignore_index=True)
    
    if comparison.shape[0] > 0:
        print("\nСравнительная таблица метрик:")
        print(comparison.to_string(index=False))
        
        # Визуализация сравнения
        if len(comparison) > 1:
            metrics_to_plot = [col for col in comparison.columns if col != 'Метод']
            fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(5*len(metrics_to_plot), 6))
            if len(metrics_to_plot) == 1:
                axes = [axes]
            
            for idx, metric in enumerate(metrics_to_plot):
                axes[idx].bar(comparison['Метод'], comparison[metric], color=['steelblue', 'orange'])
                axes[idx].set_ylabel(metric)
                axes[idx].set_title(f'Сравнение {metric}')
                axes[idx].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('laba4/comparison.png', dpi=config.DPI, bbox_inches='tight')
            plt.close()
    
    print("\n" + "="*60)
    print("РАБОТА ЗАВЕРШЕНА")
    print("="*60)

if __name__ == '__main__':
    main()

