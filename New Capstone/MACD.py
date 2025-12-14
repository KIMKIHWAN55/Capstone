import pandas as pd
import numpy as np
import random
from deap import base, creator, tools, algorithms
import talib
import multiprocessing
from eval_signal3 import calculate_total_fitness_optimized

def init_creator():
    """DEAP creator 초기화"""
    # FitnessMin: 오차(Fitness)를 최소화하는 것이 목표
    for name in ["FitnessMin", "Individual"]:
        if hasattr(creator, name):
            delattr(creator, name)
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

def generate_MACD_signals_vectorized(
    data: pd.DataFrame, 
    fast: int, 
    slow: int, 
    signal: int, 
    buy_thresh: float, 
    sell_thresh: float
) -> pd.DataFrame:

    df = data.copy()
    close_values = df['Close'].values
    
    # 1. MACD 계산 (Histogram 사용)
    # macdhist = MACD Line - Signal Line
    macd, macdsignal, macdhist = talib.MACD(
        close_values, 
        fastperiod=fast, 
        slowperiod=slow, 
        signalperiod=signal
    )
    
    df['Hist'] = macdhist
    df = df.dropna(subset=['Hist'])
    if df.empty:
        return pd.DataFrame(columns=['Index', 'Close', 'Type'])

    # 2. 신호 벡터화
    df['Hist_Shifted'] = df['Hist'].shift(1)
    df = df.dropna(subset=['Hist_Shifted'])
    if df.empty:
        return pd.DataFrame(columns=['Index', 'Close', 'Type'])

    # (a) BUY 신호: 히스토그램이 'buy_thresh'를 상향 돌파
    buy_signals = (df['Hist_Shifted'] <= buy_thresh) & (df['Hist'] > buy_thresh)
    
    # (b) SELL 신호: 히스토그램이 'sell_thresh'를 하향 돌파
    sell_signals = (df['Hist_Shifted'] >= sell_thresh) & (df['Hist'] < sell_thresh)

    # 3. 신호 취합 및 반환
    df['Type'] = np.nan
    df.loc[buy_signals, 'Type'] = 'BUY'
    df.loc[sell_signals, 'Type'] = 'SELL'

    final_signals_df = df.dropna(subset=['Type']).copy()

    if final_signals_df.empty:
        return pd.DataFrame(columns=['Index', 'Close', 'Type'])

    # 최종 포맷 맞추기 (eval_signal3.py 호환)
    final_signals_df = final_signals_df[['Close', 'Type']].copy()
    final_signals_df.reset_index(inplace=True)
    index_column_name = final_signals_df.columns[0]
    final_signals_df.rename(columns={index_column_name: 'Index'}, inplace=True)
    
    return final_signals_df

def evaluate_MACD_individual(
    individual, df_data, expected_trading_points_df, param_bounds
):

    # 5개 파라미터 언패킹
    fast, slow, signal, buy_thresh, sell_thresh = individual
    
    # --- 5개 파라미터 수리(Repair) ---
    fast = int(round(fast))
    slow = int(round(slow))
    signal = int(round(signal))
    
    fast = np.clip(fast, param_bounds['fast'][0], param_bounds['fast'][1])
    slow = np.clip(slow, param_bounds['slow'][0], param_bounds['slow'][1])
    signal = np.clip(signal, param_bounds['signal'][0], param_bounds['signal'][1])
    
    buy_thresh = np.clip(buy_thresh, 
                         param_bounds['buy_thresh'][0], 
                         param_bounds['buy_thresh'][1])
    
    sell_thresh = np.clip(sell_thresh, 
                          param_bounds['sell_thresh'][0], 
                          param_bounds['sell_thresh'][1])

    # --- 논리적 제약조건 '검증' ---
    # 1. fast는 slow보다 작아야 함 (TALIB 제약)
    if fast >= slow:
        return (float('inf'),) # 불가능한 조합이므로 최대 패널티
        
    # 2. fast, slow, signal은 2 이상이어야 함 (Bounds가 보장)

    # 파라미터로 벡터화된 함수 호출
    suggested_signals_df = generate_MACD_signals_vectorized(
        df_data, fast, slow, signal, buy_thresh, sell_thresh
    )
    
    # 'eval_signal3'를 사용한 피트니스(오차) 계산
    fitness = calculate_total_fitness_optimized(
        df_data, expected_trading_points_df, suggested_signals_df
    )
    
    if expected_trading_points_df.empty and suggested_signals_df.empty:
        return (0.0,) # 둘 다 비었으면 오차 0
        
    if fitness == float('inf'):
        return (1000000000.0,) # inf 대신 큰 값 반환

    return (fitness,)

def run_MACD_ga_optimization(
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
        raise ValueError("입력 DataFrame에 'Close' 컬럼이 반드시 포함되어야 합니다.")
    if 'cpm_point_type' not in df_data.columns:
        raise ValueError("입력 DataFrame에 'cpm_point_type' 컬럼이 반드시 포함되어야 합니다.")
        
    signal_rows = df_data.loc[df_data['cpm_point_type'] != 0].copy()
    signal_rows['Type'] = signal_rows['cpm_point_type'].map({-1: 'BUY', 1: 'SELL'})
    expected_trading_points_df = pd.DataFrame({
        'Index': signal_rows.index,
        'Type': signal_rows['Type'],
        'Close': signal_rows['Close']
    })
    
    # --- 5개 파라미터 경계값 ---
    PARAM_BOUNDS = {
        'fast': (2, 40),      # Fast EMA 기간 (최소 2 이상)
        'slow': (10, 100),    # Slow EMA 기간 (fast보다 커야 함)
        'signal': (2, 30),  # Signal EMA 기간 (최소 2 이상)
        'buy_thresh': (-2.0, 1.0),  # 매수 히스토그램 임계값
        'sell_thresh': (-1.0, 2.0), # 매도 히스토그램 임계값
    }

    toolbox = base.Toolbox()

    # --- 5개 속성(Attribute) 등록 ---
    toolbox.register("attr_fast", random.randint, *PARAM_BOUNDS['fast'])
    toolbox.register("attr_slow", random.randint, *PARAM_BOUNDS['slow'])
    toolbox.register("attr_signal", random.randint, *PARAM_BOUNDS['signal'])
    toolbox.register("attr_buy_thresh", random.uniform, *PARAM_BOUNDS['buy_thresh'])
    toolbox.register("attr_sell_thresh", random.uniform, *PARAM_BOUNDS['sell_thresh'])

    # --- 5개 파라미터 개체(Individual) 정의 ---
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_fast, toolbox.attr_slow,
                      toolbox.attr_signal, toolbox.attr_buy_thresh, 
                      toolbox.attr_sell_thresh), n=1)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # evaluate 함수에 param_bounds 전달
    toolbox.register("evaluate", evaluate_MACD_individual, 
                     df_data=df_data, 
                     expected_trading_points_df=expected_trading_points_df,
                     param_bounds=PARAM_BOUNDS)

    # 교배(Mate) 및 변이(Mutate)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, 
                     mu=[0]*5, 
                     sigma=[2, 5, 2, 0.2, 0.2], # 각 파라미터별 변이 강도
                     indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # --- 멀티프로세싱 설정 ---
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    print("--- MACD 5-Params 벡터화 GA 최적화 시작 ---")
    pop = toolbox.population(n=population_size)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    hof = tools.HallOfFame(1)

    # --- GA 실행 ---
    algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=generations,
                        stats=stats, halloffame=hof, verbose=True)

    #pool.close()
    #pool.join()

    best_individual = hof[0]
    best_fitness = best_individual.fitness.values[0]

    print("\n--- MACD 유전 알고리즘 결과 ---")
    print(f"최적 5-파라미터 (Raw): {best_individual}")
    print(f"최소 적합도 (오차): {best_fitness}")

    # --- 5-파라미터 최종 수리(Repair) ---
    final_fast = np.clip(int(round(best_individual[0])), *PARAM_BOUNDS['fast'])
    final_slow = np.clip(int(round(best_individual[1])), *PARAM_BOUNDS['slow'])
    final_signal = np.clip(int(round(best_individual[2])), *PARAM_BOUNDS['signal'])
    
    # fast >= slow 제약조건 강제 조정
    if final_fast >= final_slow:
        final_fast = final_slow - 1
        final_fast = np.clip(final_fast, *PARAM_BOUNDS['fast']) # fast 최소값 보장
        print(f"(경고) fast >= slow. fast를 {final_fast}로 강제 조정.")

    final_buy_thresh = np.clip(best_individual[3], *PARAM_BOUNDS['buy_thresh'])
    final_sell_thresh = np.clip(best_individual[4], *PARAM_BOUNDS['sell_thresh'])
    
    final_params = (final_fast, final_slow, final_signal, 
                    final_buy_thresh, final_sell_thresh)
    
    print(f"수리된 파라미터 (fast, slow, signal, buy_th, sell_th): {final_params}")

    # --- 최종 신호 생성 ---
    suggested_signals_from_best_params = generate_MACD_signals_vectorized(
        df_data, *final_params
    )

    # 'MACD_Signals' 컬럼에 최종 결과 저장
    df_data['MACD_Signals'] = 0
    if not suggested_signals_from_best_params.empty:
        signal_map = suggested_signals_from_best_params.set_index('Index')['Type'].map({'BUY': -1, 'SELL': 1})
        df_data['MACD_Signals'] = signal_map.reindex(df_data.index).fillna(0).astype(int)
    
    return final_params, best_fitness, df_data