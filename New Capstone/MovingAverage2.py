import pandas as pd
import numpy as np
import random
from deap import base, creator, tools, algorithms
import talib
import multiprocessing
from eval_signal3 import calculate_total_fitness_optimized

def init_creator():
    """DEAP creator 초기화"""
    for name in ["FitnessMin", "Individual"]:
        if hasattr(creator, name):
            delattr(creator, name)
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

def generate_MA_signals_vectorized(
    data: pd.DataFrame, 
    N: int, n: int, 
    a: float, b: float, c: float,           # 매수(BUY)용 파라미터
    a_bar: float, b_bar: float, c_bar: float # 매도(SELL)용 파라미터 (ā, ̄b, ̄c)
) -> pd.DataFrame:
    
    # 1. 지표 계산
    df = data.copy()
    close_values = df['Close'].values

    ma_N_values = talib.SMA(close_values, timeperiod=N)
    ma_n_values = talib.SMA(close_values, timeperiod=n)

    df['MA_N'] = pd.Series(ma_N_values, index=df.index)
    df['MA_n'] = pd.Series(ma_n_values, index=df.index)
    
    df['Zt'] = df['MA_N'] - df['MA_n'] # Zt = Long - Short
    df['Wk'] = -df['Zt']               # Wk = -Zt (SELL 로직용)
    
    df = df.dropna(subset=['Zt'])
    if df.empty:
        return pd.DataFrame(columns=['Index', 'Close', 'Type'])

    # 2. 크로스(Cross) 및 리짐(Regime) 그룹 정의
    # Zt의 부호를 확인
    df['Sign'] = np.sign(df['Zt'])
    
    # 부호가 바뀌는 지점(Cross)을 찾음 (0이 아닌 값)
    df['Cross'] = df['Sign'].diff().fillna(0)
    
    # 크로스가 발생할 때마다 새 그룹 ID를 부여
    # (e.g., 0, 0, 0, 1, 1, 1, 2, 2, 2, ...)
    df['RegimeGroup'] = (df['Cross'] != 0).cumsum()

    # 3. 그룹별 Expanding Max 계산 (O(n^2) -> O(n)의 핵심)
    
    # BUY 로직(MEt): Zt가 양수(+)인 값들만 대상으로 함
    df['Zt_buy'] = df['Zt'].where(df['Sign'] >= 0)
    # SELL 로직(MWk): Wk(=-Zt)가 양수(+)인 값들만 대상으로 함 (즉, Zt < 0)
    df['Wk_sell'] = df['Wk'].where(df['Sign'] < 0)

    # 각 RegimeGroup 별로, 그룹 시작부터 현재까지의 최대값을 계산
    # .expanding().max()가 이 작업을 O(n)으로 수행
    df['MEt_calc'] = df.groupby('RegimeGroup')['Zt_buy'].expanding().max().reset_index(level=0, drop=True)
    df['MWk_calc'] = df.groupby('RegimeGroup')['Wk_sell'].expanding().max().reset_index(level=0, drop=True)

    # 4. 조건절 벡터화 (Boolean Masking)

    # (a) BUY 신호 조건 (Eq 8, 9)
    cond_buy_regime = (df['Sign'] >= 0)
    cond_buy_8 = (df['MEt_calc'] > (b * c))
    # Eq 9: Zt < min(MEt_calc / a, c)
    min_val_buy = (df['MEt_calc'] / a)
    cond_buy_9 = (df['Zt'] < np.minimum(min_val_buy, c))
    
    buy_signals = cond_buy_regime & cond_buy_8 & cond_buy_9

    # (b) SELL 신호 조건 (Eq 10, 11)
    cond_sell_regime = (df['Sign'] < 0)
    cond_sell_10 = (df['MWk_calc'] > (b_bar * c_bar))
    # Eq 11: Wk < min(MWk_calc / a_bar, c_bar)
    min_val_sell = (df['MWk_calc'] / a_bar)
    cond_sell_11 = (df['Wk'] < np.minimum(min_val_sell, c_bar))
    
    sell_signals = cond_sell_regime & cond_sell_10 & cond_sell_11

    # 5. 신호 취합 및 반환
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

def evaluate_MA_individual(
    individual, df_data, expected_trading_points_df, param_bounds
):

    # 8개 파라미터 언패킹
    N, n, a, b, c, a_bar, b_bar, c_bar = individual
    
    # --- 8개 파라미터 (np.clip 사용) ---
    N = int(round(N))
    n = int(round(n))
    
    N = np.clip(N, param_bounds['N'][0], param_bounds['N'][1])
    n = np.clip(n, param_bounds['n'][0], param_bounds['n'][1])
    
    if N <= n:
        return (float('inf'),) 

    a = np.clip(a, param_bounds['a'][0], param_bounds['a'][1])
    b = np.clip(b, param_bounds['b'][0], param_bounds['b'][1])
    c = np.clip(c, param_bounds['c'][0], param_bounds['c'][1])
    
    a_bar = np.clip(a_bar, param_bounds['a_bar'][0], param_bounds['a_bar'][1])
    b_bar = np.clip(b_bar, param_bounds['b_bar'][0], param_bounds['b_bar'][1])
    c_bar = np.clip(c_bar, param_bounds['c_bar'][0], param_bounds['c_bar'][1])

    # 벡터화된 함수 호출
    suggested_signals_df = generate_MA_signals_vectorized(
        df_data, N, n, a, b, c, a_bar, b_bar, c_bar
    )
    
    fitness = calculate_total_fitness_optimized(
        df_data, expected_trading_points_df, suggested_signals_df
    )

    if expected_trading_points_df.empty and suggested_signals_df.empty:
        return (0.0,)
    
    if fitness == float('inf'):
        return (1000000000.0,) 

    return (fitness,)

def run_MA_ga_optimization(
    df_input: pd.DataFrame, 
    generations: int = 50, 
    population_size: int = 50, 
    seed: int = None
):
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    init_creator()
    
    df_data = df_input.copy()
    if isinstance(df_input, pd.Series):
        df_data = pd.DataFrame(df_input)
        df_data.columns = ['Close']
    elif 'Close' not in df_data.columns:
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

    # 파라미터 경계값 중앙 관리
    PARAM_BOUNDS = {
        'N': (41, 150),
        'n': (5, 40),
        'a': (0.1, 10.0),
        'b': (0.01, 1.0),
        'c': (0.1, 5.0),
        'a_bar': (0.1, 10.0),
        'b_bar': (0.01, 1.0),
        'c_bar': (0.1, 5.0),
    }

    toolbox = base.Toolbox()

    # 8개 파라미터 등록 (PARAM_BOUNDS 사용)
    toolbox.register("attr_N", random.randint, *PARAM_BOUNDS['N'])
    toolbox.register("attr_n", random.randint, *PARAM_BOUNDS['n'])
    toolbox.register("attr_a", random.uniform, *PARAM_BOUNDS['a'])
    toolbox.register("attr_b", random.uniform, *PARAM_BOUNDS['b'])
    toolbox.register("attr_c", random.uniform, *PARAM_BOUNDS['c'])
    toolbox.register("attr_a_bar", random.uniform, *PARAM_BOUNDS['a_bar'])
    toolbox.register("attr_b_bar", random.uniform, *PARAM_BOUNDS['b_bar'])
    toolbox.register("attr_c_bar", random.uniform, *PARAM_BOUNDS['c_bar'])

    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_N, toolbox.attr_n, 
                      toolbox.attr_a, toolbox.attr_b, toolbox.attr_c,
                      toolbox.attr_a_bar, toolbox.attr_b_bar, toolbox.attr_c_bar), n=1)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # evaluate 함수에 param_bounds 전달
    toolbox.register("evaluate", evaluate_MA_individual, 
                     df_data=df_data, 
                     expected_trading_points_df=expected_trading_points_df,
                     param_bounds=PARAM_BOUNDS) # <-- 추가

    # 8-파라미터 유전 연산자
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, 
                     mu=[0]*8, 
                     sigma=[5, 2, 0.5, 0.05, 0.1, 0.5, 0.05, 0.1], 
                     indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    print("--- MA 8-Params 벡터화 GA 최적화 시작 ---")
    pop = toolbox.population(n=population_size)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    hof = tools.HallOfFame(1)

    algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=generations,
                        stats=stats, halloffame=hof, verbose=True)

    pool.close()
    pool.join()

    best_individual = hof[0]
    best_fitness = best_individual.fitness.values[0]

    print("\n--- 이동평균 유전 알고리즘 결과 ---")
    print(f"최적의 8-파라미터: {best_individual}")
    print(f"최소 적합도: {best_fitness}")

    final_N = np.clip(int(round(best_individual[0])), *PARAM_BOUNDS['N'])
    final_n = np.clip(int(round(best_individual[1])), *PARAM_BOUNDS['n'])
    
    if final_N <= final_n:
        final_n = final_N - 1
        print(f"(경고) 최적 개체가 N <= n 이었습니다. n을 {final_n}으로 강제 조정합니다.")

    final_a = np.clip(best_individual[2], *PARAM_BOUNDS['a'])
    final_b = np.clip(best_individual[3], *PARAM_BOUNDS['b'])
    final_c = np.clip(best_individual[4], *PARAM_BOUNDS['c'])
    final_a_bar = np.clip(best_individual[5], *PARAM_BOUNDS['a_bar'])
    final_b_bar = np.clip(best_individual[6], *PARAM_BOUNDS['b_bar'])
    final_c_bar = np.clip(best_individual[7], *PARAM_BOUNDS['c_bar'])
    
    final_params = (final_N, final_n, final_a, final_b, final_c,
                    final_a_bar, final_b_bar, final_c_bar)
    
    print(f"수리된 파라미터 (N, n, a, b, c, ā, ̄b, ̄c): {final_params}")

    #벡터화된 함수 호출
    suggested_signals_from_best_params = generate_MA_signals_vectorized(
        df_data, *final_params
    )

    df_data['MA_Signals'] = 0
    if not suggested_signals_from_best_params.empty:
        signal_map = suggested_signals_from_best_params.set_index('Index')['Type'].map({'BUY': -1, 'SELL': 1})
        df_data['MA_Signals'] = signal_map.reindex(df_data.index).fillna(0).astype(int)
    
    return final_params, best_fitness, df_data