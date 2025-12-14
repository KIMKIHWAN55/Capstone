import pandas as pd
import numpy as np
import random
from deap import base, creator, tools, algorithms
import talib
import multiprocessing
from eval_signal3 import calculate_total_fitness_optimized

def init_creator():
    """DEAP creator 초기화 (FitnessMin: 오차 최소화)"""
    for name in ["FitnessMin", "Individual"]:
        if hasattr(creator, name):
            delattr(creator, name)
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

def generate_BB_signals_vectorized(
    data: pd.DataFrame, 
    period: int, 
    std_dev_upper: float, 
    std_dev_lower: float
) -> pd.DataFrame:

    df = data.copy()
    close_values = df['Close'].values
    
    # 1. 볼린저 밴드 계산
    upper, middle, lower = talib.BBANDS(
        close_values, 
        timeperiod=period, 
        nbdevup=std_dev_upper, 
        nbdevdn=std_dev_lower
    )
    
    df['Upper_Band'] = upper
    df['Lower_Band'] = lower
    df = df.dropna(subset=['Upper_Band', 'Lower_Band'])
    if df.empty:
        return pd.DataFrame(columns=['Index', 'Close', 'Type'])

    # 2. 신호 벡터화 (가격이 밴드를 '처음' 터치/돌파하는 순간)
    df['Close_Shifted'] = df['Close'].shift(1).fillna(df['Close'])
    
    # (a) BUY 신호: 가격이 위에서 아래로 하단 밴드를 하향 돌파 (또는 터치)
    buy_signals = (df['Close_Shifted'] > df['Lower_Band']) & (df['Close'] <= df['Lower_Band'])
    
    # (b) SELL 신호: 가격이 아래에서 위로 상단 밴드를 상향 돌파 (또는 터치)
    sell_signals = (df['Close_Shifted'] < df['Upper_Band']) & (df['Close'] >= df['Upper_Band'])

    # 3. 신호 취합 및 반환
    df['Type'] = np.nan
    df.loc[buy_signals, 'Type'] = 'BUY'
    df.loc[sell_signals, 'Type'] = 'SELL'

    final_signals_df = df.dropna(subset=['Type']).copy()

    if final_signals_df.empty:
        return pd.DataFrame(columns=['Index', 'Close', 'Type'])

    # 최종 포맷 맞추기
    final_signals_df = final_signals_df[['Close', 'Type']].copy()
    final_signals_df.reset_index(inplace=True)
    index_column_name = final_signals_df.columns[0]
    final_signals_df.rename(columns={index_column_name: 'Index'}, inplace=True)
    
    return final_signals_df

def evaluate_BB_individual(
    individual, df_data, expected_trading_points_df, param_bounds
):
    # 3개 파라미터 언패킹
    period, std_dev_upper, std_dev_lower = individual
    
    # --- 파라미터 수리(Repair) ---
    period = int(round(period))
    period = np.clip(period, param_bounds['period'][0], param_bounds['period'][1])
    
    std_dev_upper = np.clip(std_dev_upper, 
                            param_bounds['std_dev_upper'][0], 
                            param_bounds['std_dev_upper'][1])
    
    std_dev_lower = np.clip(std_dev_lower, 
                            param_bounds['std_dev_lower'][0], 
                            param_bounds['std_dev_lower'][1])

    # 파라미터로 벡터화된 함수 호출
    suggested_signals_df = generate_BB_signals_vectorized(
        df_data, period, std_dev_upper, std_dev_lower
    )
    
    # 'eval_signal3'를 사용한 피트니스(오차) 계산
    fitness = calculate_total_fitness_optimized(
        df_data, expected_trading_points_df, suggested_signals_df
    )
    
    if expected_trading_points_df.empty and suggested_signals_df.empty:
        return (0.0,)
        
    if fitness == float('inf'):
        return (1000000000.0,)

    return (fitness,)

def run_BB_ga_optimization(
    df_input: pd.DataFrame, 
    generations: int = 50, 
    population_size: int = 50, 
    seed: int = None
):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    init_creator()

    # --- '정답' (CPM) 데이터 준비 ---
    df_data = df_input.copy()
    if 'Close' not in df_data.columns:
        raise ValueError("DataFrame에 'Close' 컬럼이 없습니다.")
    if 'cpm_point_type' not in df_data.columns:
        raise ValueError("DataFrame에 'cpm_point_type' 컬럼이 없습니다.")
        
    signal_rows = df_data.loc[df_data['cpm_point_type'] != 0].copy()
    signal_rows['Type'] = signal_rows['cpm_point_type'].map({-1: 'BUY', 1: 'SELL'})
    expected_trading_points_df = pd.DataFrame({
        'Index': signal_rows.index,
        'Type': signal_rows['Type'],
        'Close': signal_rows['Close']
    })
    
    # --- 3개 파라미터 경계값 ---
    PARAM_BOUNDS = {
        'period': (5, 50),         # BB 기간
        'std_dev_upper': (1.5, 4.0), # 상단 밴드 표준편차
        'std_dev_lower': (1.5, 4.0), # 하단 밴드 표준편차
    }

    toolbox = base.Toolbox()

    # --- 3개 속성(Attribute) 등록 ---
    toolbox.register("attr_period", random.randint, *PARAM_BOUNDS['period'])
    toolbox.register("attr_std_upper", random.uniform, *PARAM_BOUNDS['std_dev_upper'])
    toolbox.register("attr_std_lower", random.uniform, *PARAM_BOUNDS['std_dev_lower'])

    # --- 3개 파라미터 개체(Individual) 정의 ---
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_period, toolbox.attr_std_upper, 
                      toolbox.attr_std_lower), n=1)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate_BB_individual, 
                     df_data=df_data, 
                     expected_trading_points_df=expected_trading_points_df,
                     param_bounds=PARAM_BOUNDS)

    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, 
                     mu=[0]*3, 
                     sigma=[5, 0.2, 0.2], # 각 파라미터별 변이 강도
                     indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    print("--- BollingerBands 3-Params 벡터화 GA 최적화 시작 ---")
    pop = toolbox.population(n=population_size)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    hof = tools.HallOfFame(1)

    algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=generations,
                        stats=stats, halloffame=hof, verbose=True)

    #pool.close()
    #pool.join()

    best_individual = hof[0]
    best_fitness = best_individual.fitness.values[0]

    print("\n--- 볼린저 밴드 유전 알고리즘 결과 ---")
    print(f"최적 3-파라미터 (Raw): {best_individual}")
    print(f"최소 적합도 (오차): {best_fitness}")

    # --- 3-파라미터 최종 수리(Repair) ---
    final_period = np.clip(int(round(best_individual[0])), *PARAM_BOUNDS['period'])
    final_std_upper = np.clip(best_individual[1], *PARAM_BOUNDS['std_dev_upper'])
    final_std_lower = np.clip(best_individual[2], *PARAM_BOUNDS['std_dev_lower'])
    
    final_params = (final_period, final_std_upper, final_std_lower)
    
    print(f"수리된 파라미터 (period, std_up, std_low): {final_params}")

    # --- 최종 신호 생성 ---
    suggested_signals_from_best_params = generate_BB_signals_vectorized(
        df_data, *final_params
    )

    # 'BB_Signals' 컬럼에 최종 결과 저장
    df_data['BB_Signals'] = 0
    if not suggested_signals_from_best_params.empty:
        signal_map = suggested_signals_from_best_params.set_index('Index')['Type'].map({'BUY': -1, 'SELL': 1})
        df_data['BB_Signals'] = signal_map.reindex(df_data.index).fillna(0).astype(int)
    
    return final_params, best_fitness, df_data