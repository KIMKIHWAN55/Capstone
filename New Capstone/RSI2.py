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

def generate_RSI_signals_vectorized(
    data: pd.DataFrame, 
    x: int, 
    overbought_level: float, 
    oversold_level: float, 
    p: float, 
    q: float
) -> pd.DataFrame:

    df = data.copy()
    
    # 1. RSI 계산
    df['RSI'] = talib.RSI(df['Close'].values, timeperiod=x)
    df = df.dropna(subset=['RSI'])
    if df.empty:
        return pd.DataFrame(columns=['Index', 'Close', 'Type'])

    # 2. 크로스(Cross) 및 리짐(Regime) 그룹 정의
    df['RSI_Shifted'] = df['RSI'].shift(1).fillna(50) # 50은 중립

    # (a) 과매수 진입/이탈 지점
    df['Overbought_Entry'] = (df['RSI_Shifted'] <= overbought_level) & (df['RSI'] > overbought_level)
    df['Overbought_Exit'] = (df['RSI_Shifted'] > overbought_level) & (df['RSI'] <= overbought_level)

    # (b) 과매도 진입/이탈 지점
    df['Oversold_Entry'] = (df['RSI_Shifted'] >= oversold_level) & (df['RSI'] < oversold_level)
    df['Oversold_Exit'] = (df['RSI_Shifted'] < oversold_level) & (df['RSI'] >= oversold_level)

    # (c) 모든 경계 교차 지점에서 새 그룹(Regime) 시작
    df['Cross'] = df['Overbought_Entry'] | df['Overbought_Exit'] | \
                  df['Oversold_Entry'] | df['Oversold_Exit']
    df['RegimeGroup'] = df['Cross'].cumsum()

    # 3. 그룹별 사선(Oblique Line) 계산
    
    # (a) 과매수 영역(RSI > level)에 있는 데이터만 필터링
    df_ob = df[df['RSI'] > overbought_level].copy()
    if not df_ob.empty:
        # TimeElapsed = (current_pos - start_pos)
        df_ob['TimeElapsed'] = df_ob.groupby('RegimeGroup').cumcount() 
        # StartRSI = df_filtered.loc[overbought_entry_idx, 'RSI']
        df_ob['StartRSI'] = df_ob.groupby('RegimeGroup')['RSI'].transform('first')
        # 사선 Y값 = StartRSI + p * TimeElapsed
        df_ob['ObliqueLine_Y'] = df_ob['StartRSI'] + p * df_ob['TimeElapsed']
        
        # 신호: RSI가 사선 아래로 하향 돌파
        df_ob['SellSignal'] = (df_ob['RSI'] < df_ob['ObliqueLine_Y'])
        
        # '신호 스팸' 방지: 그룹 내 첫 신호만 인정
        sell_indices = df_ob[df_ob['SellSignal']].groupby('RegimeGroup').head(1).index
    else:
        sell_indices = pd.Index([])

    # (b) 과매도 영역(RSI < level)에 있는 데이터만 필터링
    df_os = df[df['RSI'] < oversold_level].copy()
    if not df_os.empty:
        # TimeElapsed = (current_pos - start_pos)
        df_os['TimeElapsed'] = df_os.groupby('RegimeGroup').cumcount()
        # StartRSI = df_filtered.loc[oversold_entry_idx, 'RSI']
        df_os['StartRSI'] = df_os.groupby('RegimeGroup')['RSI'].transform('first')
        # 사선 Y값 = StartRSI + q * TimeElapsed
        df_os['ObliqueLine_Y'] = df_os['StartRSI'] + q * df_os['TimeElapsed']
        
        # 신호: RSI가 사선 위로 상향 돌파
        df_os['BuySignal'] = (df_os['RSI'] > df_os['ObliqueLine_Y'])
        
        # '신호 스팸' 방지: 그룹 내 첫 신호만 인정
        buy_indices = df_os[df_os['BuySignal']].groupby('RegimeGroup').head(1).index
    else:
        buy_indices = pd.Index([])

    # 4. 신호 취합 및 반환
    buy_df = df.loc[buy_indices, ['Close']].copy()
    buy_df['Type'] = 'BUY'
    
    sell_df = df.loc[sell_indices, ['Close']].copy()
    sell_df['Type'] = 'SELL'

    signals_df = pd.concat([buy_df, sell_df])
    
    if signals_df.empty:
        return pd.DataFrame(columns=['Index', 'Close', 'Type'])
        
    signals_df = signals_df.sort_index()
    final_signals_df = signals_df[['Close', 'Type']].copy()
    final_signals_df.reset_index(inplace=True)
    index_column_name = final_signals_df.columns[0]
    final_signals_df.rename(columns={index_column_name: 'Index'}, inplace=True)
        
    return final_signals_df

def evaluate_RSI_individual(
    individual, df_data, expected_trading_points_df, param_bounds
):

    # 5개 파라미터 언패킹
    x, overbought_level, oversold_level, p, q = individual
    
    # --- 5개 파라미터  ---
    x = int(round(x))
    x = np.clip(x, param_bounds['x'][0], param_bounds['x'][1])
    
    overbought_level = np.clip(overbought_level, 
                              param_bounds['overbought'][0], 
                              param_bounds['overbought'][1])
    
    oversold_level = np.clip(oversold_level, 
                            param_bounds['oversold'][0], 
                            param_bounds['oversold'][1])
    
    p = np.clip(p, param_bounds['p'][0], param_bounds['p'][1])
    q = np.clip(q, param_bounds['q'][0], param_bounds['q'][1])

    # --- 논리적 제약조건 '검증' ---
    # 1. overbought가 oversold보다 커야 함
    if overbought_level <= oversold_level:
        return (float('inf'),)
        
    # 2. p는 음수, q는 양수여야 함 (Bounds가 이미 보장하지만, 명시적 확인)
    #    (np.clip이 (-10.0, -0.1) / (0.1, 10.0)로 보장하므로 사실상 불필요)
    # if p >= 0 or q <= 0:
    #     return (float('inf'),)
    
    # 3. x는 talib.RSI 최소값 2 이상이어야 함 (Bounds가 5부터이므로 불필요)
    # if x < 2:
    #     return (float('inf'),)

    # 파라미터로 벡터화된 함수 호출
    suggested_signals_df = generate_RSI_signals_vectorized(
        df_data, x, overbought_level, oversold_level, p, q
    )
    
    fitness = calculate_total_fitness_optimized(
        df_data, expected_trading_points_df, suggested_signals_df
    )
    
    if expected_trading_points_df.empty and suggested_signals_df.empty:
        return (0.0,)
        
    if fitness == float('inf'):
        return (1000000000.0,)

    return (fitness,)

def run_RSI_ga_optimization(
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
        'x': (5, 30),             # RSI 기간
        'overbought': (70.0, 99.9), # 과매수 (100 미만)
        'oversold': (0.1, 30.0),  # 과매도 (0 초과)
        'p': (-10.0, -0.1),       # 매도 사선 기울기 (음수)
        'q': (0.1, 10.0),         # 매수 사선 기울기 (양수)
    }

    toolbox = base.Toolbox()

    toolbox.register("attr_x", random.randint, *PARAM_BOUNDS['x'])
    toolbox.register("attr_overbought", random.uniform, *PARAM_BOUNDS['overbought'])
    toolbox.register("attr_oversold", random.uniform, *PARAM_BOUNDS['oversold'])
    toolbox.register("attr_p", random.uniform, *PARAM_BOUNDS['p'])
    toolbox.register("attr_q", random.uniform, *PARAM_BOUNDS['q'])

    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_x, toolbox.attr_overbought,
                      toolbox.attr_oversold, toolbox.attr_p, toolbox.attr_q), n=1)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # evaluate 함수에 param_bounds 전달
    toolbox.register("evaluate", evaluate_RSI_individual, 
                     df_data=df_data, 
                     expected_trading_points_df=expected_trading_points_df,
                     param_bounds=PARAM_BOUNDS)

    toolbox.register("mate", tools.cxBlend, alpha=0.5)

    toolbox.register("mutate", tools.mutGaussian, 
                     mu=[0]*5, sigma=[1, 1, 1, 0.2, 0.2], indpb=0.1)

    toolbox.register("select", tools.selTournament, tournsize=3)

    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    print("--- RSI 5-Params 벡터화 GA 최적화 시작 ---")
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

    print("\n--- RSI 유전 알고리즘 결과 ---")
    print(f"최적 5-파라미터: {best_individual}")
    print(f"최소 적합도: {best_fitness}")

    # 5-파라미터 최종
    final_x = np.clip(int(round(best_individual[0])), *PARAM_BOUNDS['x'])
    final_overbought = np.clip(best_individual[1], *PARAM_BOUNDS['overbought'])
    final_oversold = np.clip(best_individual[2], *PARAM_BOUNDS['oversold'])
    
    # 최종본이 overbought <= oversold 제약조건을 위반할 경우 강제 조정
    if final_overbought <= final_oversold:
        final_oversold = final_overbought - 0.1 # 0.1만큼 강제 분리
        final_oversold = np.clip(final_oversold, *PARAM_BOUNDS['oversold'])
        print(f"(경고) overbought <= oversold. oversold를 {final_oversold:.2f}로 강제 조정.")

    final_p = np.clip(best_individual[3], *PARAM_BOUNDS['p'])
    final_q = np.clip(best_individual[4], *PARAM_BOUNDS['q'])
    
    final_params = (final_x, final_overbought, final_oversold, final_p, final_q)
    
    print(f"수리된 파라미터 (x, ob, os, p, q): {final_params}")

    # 벡터화된 함수 호출
    suggested_signals_from_best_params = generate_RSI_signals_vectorized(
        df_data, *final_params
    )

    df_data['RSI_Signals'] = 0
    if not suggested_signals_from_best_params.empty:
        signal_map = suggested_signals_from_best_params.set_index('Index')['Type'].map({'BUY': -1, 'SELL': 1})
        df_data['RSI_Signals'] = signal_map.reindex(df_data.index).fillna(0).astype(int)
    
    return final_params, best_fitness, df_data