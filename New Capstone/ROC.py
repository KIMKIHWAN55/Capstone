import pandas as pd
import numpy as np
import talib
import random
from deap import base, creator, tools, algorithms

from eval_signal1 import calculate_fitness_with_backtest

def generate_roc_signals(data, roc_period, roc_threshold):
    df_copy = data.copy()
    roc_period = int(roc_period)
    
    if roc_period < 2:
        return pd.DataFrame(columns=['Index', 'Type', 'Close'])
        
    roc_values = talib.ROC(df_copy['Close'], timeperiod=roc_period)
    df_copy['ROC'] = roc_values
    df_copy.dropna(inplace=True)

    signals = []
    position = 0
    for i in range(len(df_copy)):
        current_roc = df_copy['ROC'].iloc[i]
        if position == 0:
            if current_roc > roc_threshold:
                signals.append({'Index': df_copy.index[i], 'Type': 'BUY', 'Close': df_copy['Open'].iloc[i]})
                position = 1
        elif position == 1:
            if current_roc < -roc_threshold:
                signals.append({'Index': df_copy.index[i], 'Type': 'SELL', 'Close': df_copy['Open'].iloc[i]})
                position = 0
    return pd.DataFrame(signals)

def _repair_roc(ind):
    # ind = [roc_period, roc_threshold, atr_multiplier, tp_ratio]
    ind[0] = int(np.clip(round(ind[0]), 10, 50))
    ind[1] = np.clip(ind[1], 1.0, 10.0)
    ind[2] = float(np.clip(ind[2], 1.0, 5.0))
    ind[3] = float(np.clip(ind[3], 1.5, 5.0))
    return ind

def evaluate_roc_individual(individual, df_data):
    params = _repair_roc(list(individual))
    roc_period, roc_threshold, atr_multiplier, tp_ratio = params
        
    signals_df = generate_roc_signals(df_data, roc_period, roc_threshold)
    if signals_df.empty:
        return -1000.0,

    df_with_signals = df_data.copy()
    signal_map = signals_df.set_index('Index')['Type'].map({'BUY': -1, 'SELL': 1})
    df_with_signals['Signal'] = signal_map.fillna(0)
    
    return calculate_fitness_with_backtest(df_with_signals, atr_multiplier, tp_ratio)

def run_roc_ga_optimization(df_input, generations=25, population_size=50, seed=42):
    df_data = df_input.copy()
    
    try:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
    except Exception:
        pass
        
    toolbox = base.Toolbox()
    
    def generate_individual():
        return [
            random.randint(10, 50),   # roc_period
            random.uniform(1.0, 10.0),# roc_threshold
            random.uniform(1.0, 5.0), # atr_multiplier
            random.uniform(1.5, 5.0)  # tp_ratio
        ]

    toolbox.register("individual_raw", generate_individual)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.individual_raw)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_roc_individual, df_data=df_data)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    
    def mutate_and_repair(individual):
        tools.mutGaussian(individual, mu=0, sigma=[5, 1, 0.5, 0.5], indpb=0.2)
        return _repair_roc(individual),

    toolbox.register("mutate", mutate_and_repair)
    toolbox.register("select", tools.selTournament, tournsize=3)

    print("ROC 유전 알고리즘 실행 중 (샤프 지수 최적화)...")
    pop = toolbox.population(n=population_size)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    hof = tools.HallOfFame(1)

    algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.3, ngen=generations,
                        stats=stats, halloffame=hof, verbose=True)
    
    best_ind = hof[0]
    best_params = _repair_roc(list(best_ind))
    best_fitness = best_ind.fitness.values[0]
    
    print("\n--- ROC 유전 알고리즘 결과 ---")
    print(f"최적의 파라미터 (period, thres, atr, tp_r): {best_params}")
    print(f"최고의 피트니스 (Sharpe Ratio 기반): {best_fitness}")
    
    roc_period, roc_threshold, _, _ = best_params
    signals = generate_roc_signals(df_data, int(roc_period), roc_threshold)
    
    df_data['ROC_Signals'] = 0
    if not signals.empty:
        signal_map = signals.set_index('Index')['Type'].map({'BUY': -1, 'SELL': 1})
        df_data.loc[signal_map.index, 'ROC_Signals'] = signal_map.values

    return best_params, best_fitness, df_data