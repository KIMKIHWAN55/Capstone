import pandas as pd
import numpy as np
import talib
import random
from deap import base, creator, tools, algorithms

# 범용 백테스팅 피트니스 함수 임포트
from eval_signal1 import calculate_fitness_with_backtest

# ---------------------------
# 신호 생성 (변경 없음)
# ---------------------------
def generate_MA_signals(data, N, n, a, b, c):
    df_copy = data.copy()
    N, n = int(N), int(n)
    
    if N <= n or N < 2 or n < 2:
        return pd.DataFrame(columns=['Index', 'Type', 'Close'])
    
    close_values = df_copy['Close'].values
    df_copy['MA_N'] = talib.SMA(close_values, timeperiod=N)
    df_copy['MA_n'] = talib.SMA(close_values, timeperiod=n)
    
    df_copy['Avg_Volume'] = talib.SMA(df_copy['Volume'].astype(float).values, timeperiod=20)
    df_copy['ATR'] = talib.ATR(df_copy['High'].values, df_copy['Low'].values, df_copy['Close'].values, timeperiod=14)
    df_copy.dropna(inplace=True)

    signals = []
    position = 0

    for i in range(1, len(df_copy)):
        prev_row = df_copy.iloc[i-1]
        curr_row = df_copy.iloc[i]
        prev_ma_n, prev_ma_N = prev_row['MA_n'], prev_row['MA_N']
        curr_ma_n, curr_ma_N = curr_row['MA_n'], curr_row['MA_N']

        threshold = a / 1000 
        volume_condition = curr_row['Volume'] > (curr_row['Avg_Volume'] * b)
        volatility_condition = curr_row['ATR'] > c

        if position == 0:
            is_golden_cross = prev_ma_n < prev_ma_N and curr_ma_n > (curr_ma_N * (1 + threshold))
            if is_golden_cross and volume_condition and volatility_condition:
                signals.append({'Index': curr_row.name, 'Type': 'BUY', 'Close': curr_row['Open']})
                position = 1
        elif position == 1:
            is_dead_cross = prev_ma_n > prev_ma_N and curr_ma_n < (curr_ma_N * (1 - threshold))
            if is_dead_cross and volume_condition and volatility_condition:
                signals.append({'Index': curr_row.name, 'Type': 'SELL', 'Close': curr_row['Open']})
                position = 0
                
    return pd.DataFrame(signals)

# ---------------------------
# 경계/관계 제약 보정(Repair)
# ---------------------------
def _repair_ma(ind):
    # ind = [N, n, a, b, c, atr_multiplier, tp_ratio]
    ind[0] = int(np.clip(round(ind[0]), 10, 200))
    ind[1] = int(np.clip(round(ind[1]), 5, 50))
    
    min_gap = 10
    if ind[0] < ind[1] + min_gap:
        ind[0] = ind[1] + min_gap
        ind[0] = min(ind[0], 200)

    ind[2] = np.clip(ind[2], 0.1, 5.0)  # a
    ind[3] = np.clip(ind[3], 0.1, 5.0)  # b
    ind[4] = np.clip(ind[4], 0.1, 5.0)  # c
    ind[5] = float(np.clip(ind[5], 1.0, 5.0)) # atr_multiplier
    ind[6] = float(np.clip(ind[6], 1.5, 5.0)) # tp_ratio
    return ind

# ---------------------------
# 개체 평가 (백테스팅 기반)
# ---------------------------
def evaluate_MA_individual(individual, df_data):
    params = _repair_ma(list(individual))
    N, n, a, b, c, atr_multiplier, tp_ratio = params

    signals_df = generate_MA_signals(df_data, N, n, a, b, c)
    if signals_df.empty:
        return -1000.0,

    df_with_signals = df_data.copy()
    signal_map = signals_df.set_index('Index')['Type'].map({'BUY': -1, 'SELL': 1})
    df_with_signals['Signal'] = signal_map.fillna(0)

    return calculate_fitness_with_backtest(df_with_signals, atr_multiplier, tp_ratio)

# ---------------------------
# GA 최적화 본체
# ---------------------------
def run_MA_ga_optimization(df_input, generations=25, population_size=50, seed=42):
    df_data = df_input.copy()

    try:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
    except Exception:
        pass

    toolbox = base.Toolbox()

    def generate_individual():
        return [
            random.randint(10, 200), # N
            random.randint(5, 50),   # n
            random.uniform(0.1, 5.0),# a
            random.uniform(0.1, 5.0),# b
            random.uniform(0.1, 5.0),# c
            random.uniform(1.0, 5.0),# atr_multiplier
            random.uniform(1.5, 5.0) # tp_ratio
        ]

    toolbox.register("individual_raw", generate_individual)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.individual_raw)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_MA_individual, df_data=df_data)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    
    def mutate_and_repair(individual):
        tools.mutGaussian(individual, mu=0, sigma=[10, 5, 1, 1, 1, 0.5, 0.5], indpb=0.2)
        return _repair_ma(individual),
        
    toolbox.register("mutate", mutate_and_repair)
    toolbox.register("select", tools.selTournament, tournsize=3)

    print("이동평균 유전 알고리즘 실행 중 (샤프 지수 최적화)...")
    pop = toolbox.population(n=population_size)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    hof = tools.HallOfFame(1)

    algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.3, ngen=generations,
                        stats=stats, halloffame=hof, verbose=True)
    
    best_ind = hof[0]
    best_params = _repair_ma(list(best_ind))
    best_fitness = best_ind.fitness.values[0]

    print("\n--- 이동평균 유전 알고리즘 결과 ---")
    print(f"최적의 파라미터 (N, n, a, b, c, atr, tp_r): {best_params}")
    print(f"최고의 피트니스 (Sharpe Ratio 기반): {best_fitness}")

    final_N, final_n, final_a, final_b, final_c, _, _ = best_params
    signals = generate_MA_signals(df_data, final_N, final_n, final_a, final_b, final_c)

    df_data['MA_Signals'] = 0
    if not signals.empty:
        signal_map = signals.set_index('Index')['Type'].map({'BUY': -1, 'SELL': 1})
        df_data.loc[signal_map.index, 'MA_Signals'] = signal_map.values

    return best_params, best_fitness, df_data