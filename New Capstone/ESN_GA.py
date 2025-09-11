import pandas as pd
import numpy as np
import random
from deap import base, creator, tools, algorithms
from backtesting import Backtest, Strategy
from ESN_Signals import esn_signals
from sklearn.model_selection import TimeSeriesSplit
import warnings
import traceback
import talib

warnings.filterwarnings('ignore')

class PredictedSignalStrategy(Strategy):
    # ATR 계산을 위한 파라미터 (이 값들도 나중에 GA로 최적화 가능)
    atr_period = 14      # ATR 계산 기간 (일반적으로 14 사용)
    atr_multiplier = None # 손절매를 위한 ATR 배수 (보통 2 또는 3 사용)
    tp_ratio = None

    def init(self):
        # 기존 signal 정의는 그대로 둡니다.
        self.signal = self.I(lambda x: x, self.data.Predicted_Signals, name='signal')
        
        # ATR 지표를 계산하고 self.atr에 저장합니다.
        self.atr = self.I(talib.ATR, 
                          self.data.High, 
                          self.data.Low, 
                          self.data.Close, 
                          timeperiod=self.atr_period, 
                          name="ATR")

    def next(self):
        current_signal = self.signal[-1]
        
        if current_signal == -1 and not self.position:
            current_atr = self.atr[-1]
            
            # self.atr_multiplier는 bt.run()에서 전달된 값으로 자동 설정됩니다.
            sl_price = self.data.Close[-1] - (self.atr_multiplier * current_atr)
            
            stop_loss_distance = self.data.Close[-1] - sl_price
            
            # self.tp_ratio는 bt.run()에서 전달된 값으로 자동 설정됩니다.
            tp_price = self.data.Close[-1] + (stop_loss_distance * self.tp_ratio)
            
            self.buy(sl=sl_price, tp=tp_price)
            
        elif current_signal == 1 and self.position.is_long:
            self.position.close()

PARAM_RANGES = {
    'n_reservoir': {'min': 100, 'max': 300, 'type': int},
    'spectral_radius': {'min': 0.7, 'max': 1.2, 'type': float},
    'sparsity': {'min': 0.1, 'max': 0.7, 'type': float},
    'signal_threshold': {'min': 0.1, 'max': 0.5, 'type': float},
    # --- 추가된 파라미터 ---
    'atr_multiplier': {'min': 1.0, 'max': 5.0, 'type': float}, # 손절매 ATR 배수
    'tp_ratio': {'min': 1.5, 'max': 5.0, 'type': float}      # 익절/손절 비율
}

try:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
except RuntimeError:
    pass

def generate_individual():
    n_res = random.randint(PARAM_RANGES['n_reservoir']['min'], PARAM_RANGES['n_reservoir']['max'])
    spec_rad = random.uniform(PARAM_RANGES['spectral_radius']['min'], PARAM_RANGES['spectral_radius']['max'])
    sp = random.uniform(PARAM_RANGES['sparsity']['min'], PARAM_RANGES['sparsity']['max'])
    sig_thresh = random.uniform(PARAM_RANGES['signal_threshold']['min'], PARAM_RANGES['signal_threshold']['max'])
    # --- 추가된 파라미터 ---
    atr_multi = random.uniform(PARAM_RANGES['atr_multiplier']['min'], PARAM_RANGES['atr_multiplier']['max'])
    tp_r = random.uniform(PARAM_RANGES['tp_ratio']['min'], PARAM_RANGES['tp_ratio']['max'])
    return [n_res, spec_rad, sp, sig_thresh, atr_multi, tp_r]

def fitness_function_with_backtesting(params, train_df: pd.DataFrame, test_df: pd.DataFrame, Technical_Signals=None):
    # 1. 파라미터 언패킹 및 유효 범위 보정
    n_reservoir, spectral_radius, sparsity, signal_threshold, atr_multiplier, tp_ratio = params
    
    n_reservoir = int(round(n_reservoir))
    n_reservoir = max(PARAM_RANGES['n_reservoir']['min'], min(n_reservoir, PARAM_RANGES['n_reservoir']['max']))
    spectral_radius = max(PARAM_RANGES['spectral_radius']['min'], min(spectral_radius, PARAM_RANGES['spectral_radius']['max']))
    sparsity = max(PARAM_RANGES['sparsity']['min'], min(sparsity, PARAM_RANGES['sparsity']['max']))
    signal_threshold = max(PARAM_RANGES['signal_threshold']['min'], min(signal_threshold, PARAM_RANGES['signal_threshold']['max']))
    atr_multiplier = max(PARAM_RANGES['atr_multiplier']['min'], min(atr_multiplier, PARAM_RANGES['atr_multiplier']['max']))
    tp_ratio = max(PARAM_RANGES['tp_ratio']['min'], min(tp_ratio, PARAM_RANGES['tp_ratio']['max']))
    
    try:
        # 2. ESN 모델로 신호 생성
        backtest_signals_df = esn_signals(
            train_df=train_df, 
            test_df=test_df, 
            Technical_Signals=Technical_Signals, 
            n_reservoir=n_reservoir, 
            spectral_radius=spectral_radius, 
            sparsity=sparsity, 
            signal_threshold=signal_threshold
        )
        
        # 신호 생성이 안된 경우 페널티
        if backtest_signals_df.empty or 'Predicted_Signals' not in backtest_signals_df.columns:
            return -1000.0,
        
        # 3. 백테스팅 데이터 준비
        backtest_data = test_df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        backtest_data['Predicted_Signals'] = backtest_signals_df['Predicted_Signals'].reindex(backtest_data.index).fillna(0)
        
        # 4. 백테스팅 실행
        bt = Backtest(backtest_data, PredictedSignalStrategy, cash=10000, commission=.002, exclusive_orders=True)
        stats = bt.run(atr_multiplier=atr_multiplier, tp_ratio=tp_ratio)

        # 거래가 없는 경우 페널티
        if stats['# Trades'] == 0:
            return -100.0,

        # ======================================================================
        # 5. 피트니스 계산 (수익률 기반 + MDD 페널티)
        # ======================================================================

        # 기본 점수는 총수익률로 설정
        total_return = stats['Return [%]']

        # MDD(최대 낙폭)를 위험 관리 지표로 사용
        max_drawdown = abs(stats['Max. Drawdown [%]'])

        # 수익률에서 MDD를 직접 빼서 위험 조정 수익을 계산
        # 예: 수익률이 30%이고 MDD가 20%이면 최종 점수는 10점
        # 예: 수익률이 15%이고 MDD가 25%이면 최종 점수는 -10점
        fitness = total_return - max_drawdown

        return fitness,

    except Exception:
        # 그 외 모든 예외 발생 시 페널티
        return -1000.0,

toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, generate_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", fitness_function_with_backtesting)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
# --- 수정: sigma 리스트에 2개 값 추가 ---
sigma_vals = [
    (PARAM_RANGES['n_reservoir']['max'] - PARAM_RANGES['n_reservoir']['min']) * 0.1,
    (PARAM_RANGES['spectral_radius']['max'] - PARAM_RANGES['spectral_radius']['min']) * 0.1,
    (PARAM_RANGES['sparsity']['max'] - PARAM_RANGES['sparsity']['min']) * 0.1,
    (PARAM_RANGES['signal_threshold']['max'] - PARAM_RANGES['signal_threshold']['min']) * 0.1,
    (PARAM_RANGES['atr_multiplier']['max'] - PARAM_RANGES['atr_multiplier']['min']) * 0.1, # atr_multiplier용 sigma
    (PARAM_RANGES['tp_ratio']['max'] - PARAM_RANGES['tp_ratio']['min']) * 0.1      # tp_ratio용 sigma
]
toolbox.register("mutate", tools.mutGaussian, mu=[0]*6, sigma=sigma_vals, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=4)

def run_genetic_algorithm(train_df_ga, test_df_ga, technical_signals_list, pop_size=50, num_generations=20, cxpb=0.7, mutpb=0.4, random_seed=42):
    random.seed(random_seed)
    np.random.seed(random_seed)
    toolbox.evaluate.keywords['train_df'] = train_df_ga
    toolbox.evaluate.keywords['test_df'] = test_df_ga
    toolbox.evaluate.keywords['Technical_Signals'] = technical_signals_list
    population = toolbox.population(n=pop_size)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    hof = tools.HallOfFame(1)
    population, log = algorithms.eaSimple(population, toolbox, cxpb, mutpb, num_generations, stats=stats, halloffame=hof, verbose=True)
    best_individual = hof[0]
    return best_individual, log

def find_best_params_with_cv(df_for_tuning, technical_signals_list, n_splits=5, pop_size=30, num_generations=15, random_seed=42):
    print(f"--- {n_splits}-분할 교차 검증으로 파라미터 최적화 시작 ---")
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # ======================================================================
    # 1단계: 각 Fold에서 최적의 파라미터 '후보군' 찾기
    # ======================================================================
    candidate_params_list = []
    all_splits = list(tscv.split(df_for_tuning)) # 재사용을 위해 split 객체를 리스트로 저장

    for i, (train_index, val_index) in enumerate(all_splits):
        print(f"\n--- CV Fold {i+1}/{n_splits} (후보 파라미터 탐색) ---")
        train_cv_df = df_for_tuning.iloc[train_index]
        val_cv_df = df_for_tuning.iloc[val_index]
        
        best_params_for_fold, _ = run_genetic_algorithm(
            train_df_ga=train_cv_df, test_df_ga=val_cv_df,
            technical_signals_list=technical_signals_list,
            pop_size=pop_size, num_generations=num_generations,
            random_seed=random_seed + i
        )
        candidate_params_list.append(best_params_for_fold)
        print(f"Fold {i+1} 최적 후보: {best_params_for_fold}")

    # ======================================================================
    # 2단계: 후보 파라미터들을 모든 Fold에 교차 평가하여 가장 안정적인 파라미터 선택
    # ======================================================================
    print("\n" + "="*50)
    print("후보 파라미터 교차 평가 시작...")
    avg_scores = []
    
    for i, params in enumerate(candidate_params_list):
        scores_for_params = []
        print(f"  -> 후보 {i+1} 평가 중...")
        for j, (train_index, val_index) in enumerate(all_splits):
            train_cv_df = df_for_tuning.iloc[train_index]
            val_cv_df = df_for_tuning.iloc[val_index]
            
            # 백테스팅 실행하여 fitness 점수 계산
            fitness = fitness_function_with_backtesting(
                params, train_cv_df, val_cv_df, technical_signals_list
            )[0]
            scores_for_params.append(fitness)
        
        # 모든 Fold에 대한 평균 점수 계산
        avg_score = np.mean(scores_for_params)
        avg_scores.append(avg_score)
        print(f"  후보 {i+1}의 5개 Fold 평균 점수: {avg_score:.4f}")

    # 가장 높은 평균 점수를 가진 파라미터를 최종 선택
    best_avg_score_index = np.argmax(avg_scores)
    final_best_params = candidate_params_list[best_avg_score_index]

    print("\n" + "="*50)
    print("교차 검증 기반 파라미터 최적화 완료!")
    print(f"최종 선택된 최적 파라미터: {final_best_params}")
    print(f"(후보 {best_avg_score_index + 1}의 평균 점수가 {avg_scores[best_avg_score_index]:.4f}로 가장 높았습니다)")
    print("="*50 + "\n")
    
    return final_best_params

def esn_rolling_forward(df, technical_signals_list, n_splits_cv=5, n_splits_forward=3, pop_size=30, num_generations=15, random_seed=42, initial_train_ratio=0.7):
    train_end_idx = int(len(df) * initial_train_ratio)
    df_for_tuning = df.iloc[:train_end_idx]
    best_params = find_best_params_with_cv(
        df_for_tuning=df_for_tuning,
        technical_signals_list=technical_signals_list,
        n_splits=n_splits_cv,
        pop_size=pop_size,
        num_generations=num_generations,
        random_seed=random_seed
    )

    cv_fitness_scores = []
    fold_returns = []
    bh_returns = []
    all_fold_stats = [] 
    last_fold_signals = None
    
    splits = list(rolling_forward_split(df, n_splits_forward, initial_train_ratio=initial_train_ratio))
    if not splits:
        print("롤링 포워드 검증을 위한 데이터 분할이 생성되지 않았습니다.")
        return None, [], [], None

    print(f"\n--- {n_splits_forward}-분할 롤링 포워드 최종 성능 검증 시작 ---")
    for i, (train_df, val_df) in enumerate(splits):
        print(f"\n--- Forward Fold {i+1}/{n_splits_forward} ---")
        try:
            stats, signals = perform_final_backtest(
                train_df=train_df, test_df=val_df, best_params=best_params,
                technical_signals_list=technical_signals_list, fold_num=("last" if i == n_splits_forward - 1 else None),
                random_state=random_seed
            )
            if stats is not None:
                all_fold_stats.append(stats) 
                
                return_percent = stats['Return [%]']
                max_drawdown = abs(stats['Max. Drawdown [%]'])
                mdd_penalty = 0
                if max_drawdown > 30: mdd_penalty = 50
                elif max_drawdown > 20: mdd_penalty = 25
                elif max_drawdown > 10: mdd_penalty = 10
                fitness_value = return_percent - mdd_penalty
                
                cv_fitness_scores.append(fitness_value)
                fold_returns.append(return_percent)
                bh_returns.append(stats['Buy & Hold Return [%]'])

                if i == n_splits_forward - 1:
                    last_fold_signals = signals
        except Exception as e:
            print(f"폴드 {i+1} 백테스팅 중 오류 발생: {e}")
            traceback.print_exc()
            
    print("\n" + "="*50)
    print("롤링 포워드 교차 검증 최종 결과 요약:")
    # ... (요약 출력 부분은 동일) ...
    print("="*50)
    
    return best_params, fold_returns, all_fold_stats, last_fold_signals

def perform_final_backtest(train_df, test_df, best_params, technical_signals_list, random_state=42, fold_num=None):
    n_reservoir, spectral_radius, sparsity, signal_threshold, atr_multiplier, tp_ratio = best_params
    n_reservoir = int(round(n_reservoir))
    n_reservoir = max(PARAM_RANGES['n_reservoir']['min'], min(n_reservoir, PARAM_RANGES['n_reservoir']['max']))
    spectral_radius = max(PARAM_RANGES['spectral_radius']['min'], min(spectral_radius, PARAM_RANGES['spectral_radius']['max']))
    sparsity = max(PARAM_RANGES['sparsity']['min'], min(sparsity, PARAM_RANGES['sparsity']['max']))
    signal_threshold = max(PARAM_RANGES['signal_threshold']['min'], min(signal_threshold, PARAM_RANGES['signal_threshold']['max']))
    final_backtest_signals_df = esn_signals(train_df=train_df, test_df=test_df, Technical_Signals=technical_signals_list, n_reservoir=n_reservoir, spectral_radius=spectral_radius, sparsity=sparsity, signal_threshold=signal_threshold, random_state=random_state)
    if not isinstance(final_backtest_signals_df, pd.DataFrame) or final_backtest_signals_df.empty or 'Predicted_Signals' not in final_backtest_signals_df.columns:
        print("ESN 모델에서 유효한 신호가 생성되지 않았습니다. 백테스팅을 건너뜀.")
        return None, None
    final_backtest_data = test_df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    final_backtest_data['Predicted_Signals'] = final_backtest_signals_df['Predicted_Signals'].fillna(0)
    bt_final = Backtest(final_backtest_data, PredictedSignalStrategy, cash=10000, commission=.002, exclusive_orders=True)
    stats_final = bt_final.run(atr_multiplier=atr_multiplier, tp_ratio=tp_ratio)
    print("\n백테스팅 결과:")
    print(stats_final)
    if fold_num is not None and fold_num == "last":
        bt_final.plot(filename='final_fold_backtest_results', open_browser=False)
    return stats_final, final_backtest_signals_df

def rolling_forward_split(df: pd.DataFrame, n_splits: int, initial_train_ratio: float = 0.5):
    total_len = len(df)
    initial_train_size = int(total_len * initial_train_ratio)
    remaining_len = total_len - initial_train_size
    if n_splits == 0:
        val_size = 0
    else:
        val_size = remaining_len // n_splits
    for i in range(n_splits):
        train_end_idx = initial_train_size + i * val_size
        val_end_idx = train_end_idx + val_size
        train_df = df.iloc[:train_end_idx].copy()
        val_df = df.iloc[train_end_idx:val_end_idx].copy()
        if val_df.empty:
            continue
        yield train_df, val_df