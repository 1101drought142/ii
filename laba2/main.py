"""
Лабораторная работа 2: Ассоциативные правила
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
import networkx as nx
import config

# Создание папки для графиков
os.makedirs('laba2', exist_ok=True)

# Настройка визуализации
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = config.FIG_SIZE
plt.rcParams['figure.dpi'] = config.DPI

def load_data():
    """Загрузка данных"""
    df = pd.read_csv(config.DATA_PATH, header=None)
    print("Форма данных:", df.shape)
    return df

def preprocess_data(df):
    """Преобразование данных в формат транзакций"""
    transactions = []
    for idx, row in df.iterrows():
        transaction = [item for item in row.values if pd.notna(item) and str(item).strip()]
        if transaction:
            transactions.append(transaction)
    
    print(f"Количество транзакций: {len(transactions)}")
    return transactions

def analyze_transactions(transactions):
    """Анализ транзакций"""
    # Длины транзакций
    transaction_lengths = [len(t) for t in transactions]
    
    plt.figure(figsize=(10, 6))
    plt.hist(transaction_lengths, bins=range(1, max(transaction_lengths) + 2), 
             edgecolor='black', alpha=0.7)
    plt.xlabel('Длина транзакции')
    plt.ylabel('Частота')
    plt.title('Распределение длин транзакций')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('laba2/transaction_lengths.png', dpi=config.DPI, bbox_inches='tight')
    plt.close()
    
    # Уникальные товары
    unique_items = set()
    for transaction in transactions:
        unique_items.update(transaction)
    
    print(f"Количество уникальных товаров: {len(unique_items)}")
    print(f"Средняя длина транзакции: {np.mean(transaction_lengths):.2f}")
    print(f"Медианная длина транзакции: {np.median(transaction_lengths):.2f}")
    
    return unique_items, transaction_lengths

def encode_transactions(transactions):
    """Преобразование транзакций в бинарный формат"""
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    print(f"Бинарная матрица: {df.shape}")
    return df

def find_frequent_itemsets_apriori(data, min_support):
    """Поиск частых наборов алгоритмом Apriori"""
    frequent_itemsets = apriori(data, min_support=min_support, use_colnames=True)
    return frequent_itemsets

def find_frequent_itemsets_fpgrowth(data, min_support):
    """Поиск частых наборов алгоритмом FPGrowth"""
    frequent_itemsets = fpgrowth(data, min_support=min_support, use_colnames=True)
    return frequent_itemsets

def generate_rules(frequent_itemsets, metric='confidence', min_threshold=0.4):
    """Генерация ассоциативных правил"""
    rules = association_rules(frequent_itemsets, metric=metric, min_threshold=min_threshold)
    return rules

def analyze_rules(rules):
    """Анализ правил"""
    if len(rules) == 0:
        print("Правила не найдены")
        return
    
    print(f"\nНайдено правил: {len(rules)}")
    print(f"\nТоп-10 правил по лифту:")
    top_lift = rules.nlargest(10, 'lift')
    print(top_lift[['antecedents', 'consequents', 'support', 'confidence', 'lift']].to_string())
    
    print(f"\nТоп-10 правил по достоверности:")
    top_confidence = rules.nlargest(10, 'confidence')
    print(top_confidence[['antecedents', 'consequents', 'support', 'confidence', 'lift']].to_string())
    
    # Анализ тривиальных правил
    high_confidence_low_lift = rules[(rules['confidence'] > 0.7) & (rules['lift'] < 1.2)]
    print(f"\nПравила с высокой достоверностью, но низким лифтом (возможно тривиальные): {len(high_confidence_low_lift)}")
    
    return top_lift, top_confidence

def visualize_top_products(frequent_itemsets):
    """Визуализация топ продуктов"""
    df_viz = frequent_itemsets.copy()
    df_viz['itemsets'] = df_viz['itemsets'].apply(lambda x: ', '.join(list(x)))
    top_products = df_viz.sort_values(by='support', ascending=False).head(config.TOP_PRODUCTS_COUNT)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='support', y='itemsets', data=top_products, palette='viridis')
    plt.title(f'Топ-{config.TOP_PRODUCTS_COUNT} самых популярных продуктов')
    plt.xlabel('Поддержка')
    plt.ylabel('Продукты')
    plt.tight_layout()
    plt.savefig('laba2/top_products.png', dpi=config.DPI, bbox_inches='tight')
    plt.close()

def visualize_rules_confidence(rules):
    """Визуализация достоверности правил"""
    if len(rules) == 0:
        return
    
    rules_viz = rules.copy()
    rules_viz['antecedents'] = rules_viz['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules_viz['consequents'] = rules_viz['consequents'].apply(lambda x: ', '.join(list(x)))
    rules_viz['rule'] = rules_viz['antecedents'] + ' -> ' + rules_viz['consequents']
    
    # Ограничиваем количество для визуализации
    top_rules = rules_viz.nlargest(20, 'confidence')
    
    plt.figure(figsize=(14, 8))
    plt.barh(range(len(top_rules)), top_rules['confidence'], color='steelblue')
    plt.yticks(range(len(top_rules)), top_rules['rule'], fontsize=8)
    plt.xlabel('Достоверность')
    plt.title('Топ-20 правил по достоверности')
    plt.tight_layout()
    plt.savefig('laba2/rules_confidence.png', dpi=config.DPI, bbox_inches='tight')
    plt.close()

def visualize_rules_graph(rules):
    """Визуализация правил в виде графа"""
    if len(rules) == 0:
        return
    
    G = nx.DiGraph()
    
    for idx, row in rules.iterrows():
        ant = ', '.join(list(row['antecedents']))
        cons = ', '.join(list(row['consequents']))
        G.add_edge(ant, cons, weight=row['confidence'], lift=row['lift'])
    
    if len(G.nodes()) == 0:
        return
    
    # Ограничиваем граф для читаемости
    if len(G.nodes()) > 30:
        # Берем топ правила по лифту
        top_rules = rules.nlargest(30, 'lift')
        G = nx.DiGraph()
        for idx, row in top_rules.iterrows():
            ant = ', '.join(list(row['antecedents']))
            cons = ', '.join(list(row['consequents']))
            G.add_edge(ant, cons, weight=row['confidence'], lift=row['lift'])
    
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    plt.figure(figsize=(16, 12))
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=1500, alpha=0.7)
    nx.draw_networkx_labels(G, pos, font_size=7)
    
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    nx.draw_networkx_edges(G, pos, width=[w*3 for w in weights], 
                          alpha=0.5, edge_color='gray', arrows=True, arrowsize=20)
    
    edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=6)
    
    plt.title('Сетевой граф ассоциативных правил')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('laba2/rules_graph.png', dpi=config.DPI, bbox_inches='tight')
    plt.close()

def custom_visualization(rules):
    """Собственная визуализация правил"""
    if len(rules) == 0:
        return
    
    # Scatter plot: confidence vs lift
    plt.figure(figsize=(10, 8))
    plt.scatter(rules['confidence'], rules['lift'], 
               s=rules['support']*1000, alpha=0.6, c=rules['support'], cmap='viridis')
    plt.colorbar(label='Поддержка')
    plt.xlabel('Достоверность')
    plt.ylabel('Лифт')
    plt.title('Визуализация правил: Достоверность vs Лифт (размер = поддержка)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('laba2/custom_visualization.png', dpi=config.DPI, bbox_inches='tight')
    plt.close()

def experiment_with_parameters(data):
    """Эксперименты с параметрами"""
    print("\n" + "="*60)
    print("ЭКСПЕРИМЕНТЫ С ПАРАМЕТРАМИ")
    print("="*60)
    
    results = []
    
    for min_support in config.SUPPORT_VALUES:
        frequent_itemsets = fpgrowth(data, min_support=min_support, use_colnames=True)
        rules = generate_rules(frequent_itemsets, min_threshold=config.MIN_CONFIDENCE)
        
        results.append({
            'min_support': min_support,
            'frequent_itemsets': len(frequent_itemsets),
            'rules': len(rules),
            'avg_confidence': rules['confidence'].mean() if len(rules) > 0 else 0,
            'avg_lift': rules['lift'].mean() if len(rules) > 0 else 0
        })
    
    results_df = pd.DataFrame(results)
    print("\nВлияние min_support:")
    print(results_df.to_string(index=False))
    
    # Визуализация
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].plot(results_df['min_support'], results_df['frequent_itemsets'], 'o-')
    axes[0, 0].set_xlabel('Min Support')
    axes[0, 0].set_ylabel('Количество частых наборов')
    axes[0, 0].set_title('Влияние min_support на количество наборов')
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(results_df['min_support'], results_df['rules'], 'o-', color='orange')
    axes[0, 1].set_xlabel('Min Support')
    axes[0, 1].set_ylabel('Количество правил')
    axes[0, 1].set_title('Влияние min_support на количество правил')
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(results_df['min_support'], results_df['avg_confidence'], 'o-', color='green')
    axes[1, 0].set_xlabel('Min Support')
    axes[1, 0].set_ylabel('Средняя достоверность')
    axes[1, 0].set_title('Влияние min_support на среднюю достоверность')
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(results_df['min_support'], results_df['avg_lift'], 'o-', color='red')
    axes[1, 1].set_xlabel('Min Support')
    axes[1, 1].set_ylabel('Средний лифт')
    axes[1, 1].set_title('Влияние min_support на средний лифт')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('laba2/parameter_experiments.png', dpi=config.DPI, bbox_inches='tight')
    plt.close()
    
    return results_df

def find_min_support_for_itemsets(data):
    """Определение минимальной поддержки для наборов разной длины"""
    print("\n" + "="*60)
    print("ОПРЕДЕЛЕНИЕ МИНИМАЛЬНОЙ ПОДДЕРЖКИ")
    print("="*60)
    
    support_values = np.arange(0.01, 0.1, 0.01)
    results = []
    
    for min_support in support_values:
        frequent_itemsets = fpgrowth(data, min_support=min_support, use_colnames=True)
        
        if len(frequent_itemsets) > 0:
            itemset_lengths = frequent_itemsets['itemsets'].apply(lambda x: len(x))
            max_length = itemset_lengths.max()
            
            for length in range(1, int(max_length) + 1):
                count = (itemset_lengths == length).sum()
                if count > 0:
                    results.append({
                        'min_support': min_support,
                        'itemset_length': length,
                        'count': count
                    })
    
    if results:
        results_df = pd.DataFrame(results)
        print("\nМинимальная поддержка для наборов разной длины:")
        pivot = results_df.pivot(index='min_support', columns='itemset_length', values='count')
        print(pivot.to_string())
        
        # Визуализация
        plt.figure(figsize=(12, 8))
        for length in sorted(results_df['itemset_length'].unique()):
            subset = results_df[results_df['itemset_length'] == length]
            plt.plot(subset['min_support'], subset['count'], 'o-', label=f'Длина {length}')
        plt.xlabel('Min Support')
        plt.ylabel('Количество наборов')
        plt.title('Зависимость количества наборов от поддержки')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('laba2/min_support_analysis.png', dpi=config.DPI, bbox_inches='tight')
        plt.close()

def main():
    """Основная функция"""
    print("="*60)
    print("ЛАБОРАТОРНАЯ РАБОТА 2: АССОЦИАТИВНЫЕ ПРАВИЛА")
    print("="*60)
    
    # Загрузка данных
    df = load_data()
    
    # Предобработка
    transactions = preprocess_data(df)
    
    # Анализ транзакций
    unique_items, transaction_lengths = analyze_transactions(transactions)
    
    # Кодирование
    data = encode_transactions(transactions)
    
    # Алгоритм FPGrowth
    print("\n" + "="*60)
    print("АЛГОРИТМ FPGROWTH")
    print("="*60)
    frequent_itemsets_fp = find_frequent_itemsets_fpgrowth(data, config.MIN_SUPPORT)
    print(f"Найдено частых наборов: {len(frequent_itemsets_fp)}")
    visualize_top_products(frequent_itemsets_fp)
    
    rules_fp = generate_rules(frequent_itemsets_fp, min_threshold=config.MIN_CONFIDENCE)
    analyze_rules(rules_fp)
    visualize_rules_confidence(rules_fp)
    visualize_rules_graph(rules_fp)
    custom_visualization(rules_fp)
    
    # Алгоритм Apriori
    print("\n" + "="*60)
    print("АЛГОРИТМ APRIORI")
    print("="*60)
    frequent_itemsets_ap = find_frequent_itemsets_apriori(data, config.MIN_SUPPORT)
    print(f"Найдено частых наборов: {len(frequent_itemsets_ap)}")
    
    rules_ap = generate_rules(frequent_itemsets_ap, min_threshold=config.MIN_CONFIDENCE)
    analyze_rules(rules_ap)
    
    # Эксперименты
    experiment_with_parameters(data)
    find_min_support_for_itemsets(data)
    
    print("\n" + "="*60)
    print("РАБОТА ЗАВЕРШЕНА")
    print("="*60)

if __name__ == '__main__':
    main()

