# Настройки для лабораторной работы 1

# Путь к данным
DATA_PATH = 'data.csv'

# Разделение данных
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Параметры моделей
RIDGE_ALPHA = 1.0
CV_FOLDS = 5

# PCA
PCA_VARIANCE_THRESHOLD = 0.95  # Доля объясненной дисперсии

# Визуализация
FIG_SIZE = (12, 8)
DPI = 100

# Метрики для расчета
METRICS = ['RMSE', 'R2', 'MAPE']

