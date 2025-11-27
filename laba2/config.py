# Настройки для лабораторной работы 2

# Путь к данным
DATA_PATH = 'Market_Basket_Optimisation.csv'

# Параметры алгоритмов
MIN_SUPPORT = 0.03
MIN_CONFIDENCE = 0.4
MIN_LIFT = 1.0

# Эксперименты с параметрами
SUPPORT_VALUES = [0.01, 0.02, 0.03, 0.04, 0.05]
CONFIDENCE_VALUES = [0.2, 0.3, 0.4, 0.5, 0.6]

# Визуализация
FIG_SIZE = (12, 8)
DPI = 100
TOP_PRODUCTS_COUNT = 10

# Граф
GRAPH_LAYOUT = 'spring'  # spring, circular, kamada_kawai

