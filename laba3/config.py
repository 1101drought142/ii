# Настройки для лабораторной работы 3

# Путь к данным
DATA_PATH = 'Iris.csv'

# Разделение данных
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Кросс-валидация
CV_FOLDS = 5

# Нейронная сеть
EPOCHS = 100
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2

# Grid Search (опционально)
LEARNING_RATES = [0.001, 0.01, 0.1]
HIDDEN_LAYERS_CONFIGS = [
    [64],
    [64, 32],
    [128, 64, 32]
]

# TensorBoard
TENSORBOARD_DIR = 'logs'

# Визуализация
FIG_SIZE = (12, 8)
DPI = 100

