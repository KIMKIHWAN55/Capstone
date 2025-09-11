import pandas as pd
import numpy as np
import talib
import random
from deap import base, creator, tools, algorithms

# --- 수정: 범용 백테스팅 피트니스 함수 임포트 ---
from eval_signal1 import calculate_fitness_with_backtest


# ---------------------------
# 신호 생성 (변경 없음)
# ---------------------------
def generate_RSI_signals(data, timeperiod, overbought_level, oversold_level):
    """
    RSI가 극단적인 구간에서 복귀할 때 매수/매도 신호를 생성합니다.
    """
    df_copy = data.copy()
    rsi_values = talib.RSI(df_copy['Close'].values, timeperiod=int(timeperiod))
    df_copy['RSI'] = pd.Series(rsi_values, index=df_copy.index)
    df_copy.dropna(inplace=True)

    signals = []
    position = 0

    for i in range(1, len(df_copy)):
        prev_rsi = df_copy['RSI'].iloc[i-1]
        curr_rsi = df_copy['RSI'].iloc[i]
        curr_row = df_copy.iloc[i]

        if position == 0 and prev_rsi < oversold_level and curr_rsi >= oversold_level:
            signals.append({'Index': curr_row.name, 'Type': 'BUY', 'Close': curr_row['Open']})
            position = 1
        
        elif position == 1 and prev_rsi > overbought_level and curr_rsi <= overbought_level:
            signals.append({'Index': curr_row.name, 'Type': 'SELL', 'Close': curr_row['Open']})
            position = 0
            
    return pd.DataFrame(signals)


# ---------------------------
# 경계/관계 제약 보정(Repair)
# ---------------------------
def _repair_and_cast(ind):
    """
    ind = [timeperiod, overbought, oversold, atr_multiplier, tp_ratio]
    - 5개의 파라미터에 대한 하드 경계 및 관계 제약을 보장합니다.
    """
    # 하드 경계
    ind[0] = int(np.clip(int(round(ind[0])), 7, 30))         # timeperiod
    ind[1] = float(np.clip(ind[1], 60.0, 80.0))              # overbought
    ind[2] = float(np.clip(ind[2], 25.0, 44.0))              # oversold
    ind[3] = float(np.clip(ind[3], 1.0, 5.0))                # atr_multiplier
    ind[4] = float(np.clip(ind[4], 1.5, 5.0))                # tp_ratio

    # 관계 제약
    margin = 20.0
    if ind[2] >= ind[1] - margin:
        mid = (ind[1] + ind[2]) / 2.0
        ind[2] = min(mid - margin / 2.0, ind[1] - margin)
        ind[2] = float(max(5.0, ind[2]))

    return ind


# ---------------------------
# 개체 평가 (백테스팅 기반)
# ---------------------------
def evaluate_RSI_individual(individual, df_data):
    """
    - 파라미터로 RSI 신호를 생성합니다.
    - 생성된 신호로 백테스트를 실행하여 샤프 지수 기반의 피트니스를 계산합니다.
    """
    # 1. 파라미터 보정
    params = _repair_and_cast(list(individual))
    timeperiod, overbought, oversold, atr_multiplier, tp_ratio = params

    # 2. 신호 생성
    signals_df = generate_RSI_signals(df_data, timeperiod, overbought, oversold)
    if signals_df.empty:
        return -1000.0,

    # 3. 데이터에 신호 결합 ('Signal' 컬럼 추가)
    df_with_signals = df_data.copy()
    signal_map = signals_df.set_index('Index')['Type'].map({'BUY': -1, 'SELL': 1})
    df_with_signals['Signal'] = signal_map.fillna(0)

    # 4. 범용 피트니스 함수 호출
    return calculate_fitness_with_backtest(df_with_signals, atr_multiplier, tp_ratio)


# ---------------------------
# GA 최적화 본체
# ---------------------------
def run_RSI_ga_optimization(df_input, generations=25, population_size=50, seed=42):
    df_data = df_input.copy()

    # --- 수정: creator를 FitnessMax로 변경 (샤프 지수는 높을수록 좋음) ---
    try:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
    except Exception:
        pass

    toolbox = base.Toolbox()

    # --- 수정: 5개 파라미터 생성 로직 ---
    def generate_individual():
        return [
            random.randint(7, 30),      # timeperiod
            random.uniform(55.0, 90.0), # overbought
            random.uniform(10.0, 45.0), # oversold
            random.uniform(1.0, 5.0),   # atr_multiplier
            random.uniform(1.5, 5.0)    # tp_ratio
        ]

    toolbox.register("individual_raw", generate_individual)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.individual_raw)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # --- 수정: evaluate 함수에서 expected_trading_points 제거 ---
    toolbox.register("evaluate", evaluate_RSI_individual, df_data=df_data)
    
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    
    # --- 수정: mutate 로직을 5개 파라미터에 맞게 변경 ---
    def mutate_and_repair(individual):
        tools.mutGaussian(individual, mu=0, sigma=[3, 5, 5, 0.5, 0.5], indpb=0.2)
        return _repair_and_cast(individual),

    toolbox.register("mutate", mutate_and_repair)
    toolbox.register("select", tools.selTournament, tournsize=3)

    print("RSI 유전 알고리즘 실행 중 (샤프 지수 최적화)...")
    pop = toolbox.population(n=population_size)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    hof = tools.HallOfFame(1)

    algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.3, ngen=generations,
                        stats=stats, halloffame=hof, verbose=True)
    
    best_ind = hof[0]
    best_params = _repair_and_cast(list(best_ind))
    best_fitness = best_ind.fitness.values[0]

    print("\n--- RSI 유전 알고리즘 결과 ---")
    print(f"최적의 파라미터 (tp, ob, os, atr, tp_r): {best_params}")
    print(f"최고의 피트니스 (Sharpe Ratio 기반): {best_fitness}")

    # 최종적으로 최적 파라미터로 생성된 신호를 데이터프레임에 추가
    tp, ob, os, _, _ = best_params
    suggested_signals_from_best = generate_RSI_signals(df_data, tp, ob, os)

    df_data['RSI_Signals'] = 0
    if not suggested_signals_from_best.empty:
        signal_map = suggested_signals_from_best.set_index('Index')['Type'].map({'BUY': -1, 'SELL': 1})
        df_data.loc[signal_map.index, 'RSI_Signals'] = signal_map.values

    return best_params, best_fitness, df_data