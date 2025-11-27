# Настройки для лабораторной работы 4

# Путь к данным
DATA_PATH = 'Iris.csv'

# Разделение данных (для внешних метрик, если есть классы)
TEST_SIZE = 0.2
RANDOM_STATE = 42

# K-means
K_RANGE = range(2, 11)  # Диапазон для подбора k
N_INIT = 10
MAX_ITER = 300

# Иерархическая кластеризация
LINKAGE_METHODS = ['ward', 'complete', 'average', 'single']
N_CLUSTERS_RANGE = range(2, 11)

# DBSCAN (опционально)
EPS_VALUES = [0.3, 0.5, 0.7, 1.0, 1.5]
MIN_SAMPLES_VALUES = [3, 5, 7, 10]

# Gaussian Mixture (опционально)
N_COMPONENTS_RANGE = range(2, 11)

# Методы кластеризации для использования
USE_KMEANS = True
USE_HIERARCHICAL = True
USE_DBSCAN = False
USE_GAUSSIAN_MIXTURE = False

# PCA для визуализации
PCA_COMPONENTS = 2

# Визуализация
FIG_SIZE = (12, 8)
DPI = 100

