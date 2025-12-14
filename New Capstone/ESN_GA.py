# 파일명: ESN_GA.py (수정완료: 수수료 고정, 변동성 필터 완화)

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

# ==================================================
# 1. 백테스팅 전략 클래스 (CV_ESN3.py와 로직 통일)
# ==================================================
class PredictedSignalStrategy(Strategy):
    # --- GA가 최적화할 파라미터들 (기본값 유지) ---
    atr_period     = 14    
    atr_multiplier = 2.0   
    tp_ratio       = 1.5   
    min_hold       = 3     
    cooldown       = 2     
    risk_per_trade = 0.05  

    def init(self):
        # 1. ESN 신호
        self.signal = self.I(lambda x: x, self.data.Predicted_Signals, name='signal')
        
        # 2. ATR (변동성)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close,
                          timeperiod=int(self.atr_period), name="ATR")
        
        # 3. SMA 120 (장기 추세선)
        self.sma_trend = self.I(talib.SMA, self.data.Close, timeperiod=120, name="SMA120")

        
        self._cd = 0  
        self.entry_bar_index = -1 
        self.initial_sl_mult = float(self.atr_multiplier) 
        self.trailing_atr_mult = float(self.tp_ratio) 
        self.current_trailing_sl_price = 0.0

    def next(self):
        # 데이터 안전 확보
        atr_now = float(self.atr[-1])
        price = float(self.data.Close[-1])
        if not np.isfinite(atr_now) or atr_now <= 0: return
        
        # 지표 값 가져오기
        sma_val = self.sma_trend[-1] if len(self.sma_trend) > 0 and np.isfinite(self.sma_trend[-1]) else price
        
        # RSI 가져오기
        try:
            rsi_val = self.data.RSI_Value[-1]
        except AttributeError:
            rsi_val = 50 

        # Avg ATR 계산
        if len(self.atr) >= 100:
            recent_atr = self.atr[-100:]
            avg_atr_val = np.mean(recent_atr[np.isfinite(recent_atr)])
        else:
            avg_atr_val = atr_now

        if self._cd > 0: self._cd -= 1
        sig = int(self.signal[-1]) 

        # ============================================================
        # [수정됨] 진입/청산 로직 단순화 (Logic Conflict Resolved)
        # ============================================================
        
        # [수정 1] 진입 로직: OR 조건 제거
        # 기존: (ESN 매수) OR (강한 추세 & 눌림목) -> 강제 진입 위험
        # 변경: ESN이 매수 신호(-1)를 보내고 + 가격이 장기 추세(SMA120) 위에 있을 때만 진입
        #       -> ESN의 예측을 최우선으로 하되, 역추세 진입만 필터링
        should_enter = (sig == -1) and (price > sma_val)

        # [수정 2] 청산 로직: 경직된 조건 제거
        # 기존: (ESN 매도) AND (추세가 강하지 않음) -> 급락장에서 청산 지연 위험
        # 변경: ESN이 매도 신호(1)를 보내면, 추세 강도와 상관없이 즉시 청산 시도
        #       -> 리스크 관리 우선
        exit_signal = (sig == 1)

        # --- [매수 실행] ---
        if should_enter and not self.position and self._cd == 0:
            
            # 1. 변동성 필터
            cutoff_multiplier = 5.0 if (price > sma_val) else 2.0
            if atr_now > (avg_atr_val * cutoff_multiplier) and rsi_val >= 70:
                return 

            # 2. 초기 손절(SL) 설정
            sl_price = price - (self.initial_sl_mult * atr_now)
            sl_dist = price - sl_price
            
            # 3. 리스크(비중) 조절
            current_risk = float(self.risk_per_trade)
            
            # (참고) 기존의 역추세 진입 비중 축소 로직은 should_enter에 price > sma_val 조건이 
            # 포함되면서 사실상 실행되지 않지만, 안전장치로 둡니다.
            # if price < sma_val: current_risk *= 0.5

            # 4. 포지션 사이징 및 진입
            equity = float(self.equity)
            if sl_dist > 0:
                size = int(max(1, min(int((equity * current_risk) / sl_dist), int((equity * 0.99) / price))))
            else:
                size = 1 # 예외 처리
            
            if size * price > equity: return
            self.buy(size=size, sl=sl_price)
            
            self.entry_bar_index = len(self.data) - 1
            self.current_trailing_sl_price = sl_price

# --- [청산 관리 (트레일링 스톱)] ---
        elif self.position and self.position.is_long:
            position_age = len(self.data) - 1 - self.entry_bar_index
            if self.entry_bar_index == -1: position_age = 0

            # [수정] 간소화된 청산 로직 적용
            if exit_signal and position_age >= int(self.min_hold):
                self.position.close()
                self._cd = int(self.cooldown)
                return

            # ============================================================
            # [추가] 변동성에 따른 동적 트레일링 스톱 보정 로직
            # ============================================================
            
            # 1. ATR 비율(현재가 대비 변동성) 계산
            # (예: 주가 100불, ATR 0.4불 -> 0.004 (0.4%))
            volatility_ratio = atr_now / price 

            # 2. 변동성에 따른 승수(effective_tp_ratio) 조정
            if volatility_ratio < 0.025: 
                # 변동성이 매우 낮으면(0.5% 미만, 예: KO), 
                # 이익 실현 목표를 2.5배 이하로 강제하여 빨리 챙김
                effective_tp_ratio = min(self.trailing_atr_mult, 3.5) 
            else:
                # 변동성이 크면(테슬라 등), 원래 GA가 찾은 값(길게) 유지
                effective_tp_ratio = self.trailing_atr_mult

            # 3. 새로운 손절가(Trailing Stop) 계산 적용
            new_sl = price - (effective_tp_ratio * atr_now)
            
            # ============================================================

            # 트레일링 스톱 갱신 (가격이 올라가면 SL도 따라 올라감)
            if new_sl > self.current_trailing_sl_price:
                self.current_trailing_sl_price = new_sl
                self.position.sl = self.current_trailing_sl_price

# =========================
# GA 탐색 파라미터 공간 (12개로 축소)
# =========================
PARAM_RANGES = {
    # ESN (5개)
    'spectral_radius': {'min': 0.80, 'max': 0.99, 'type': float},
    'sparsity':        {'min': 0.05, 'max': 0.80, 'type': float},
    'th_buy':          {'min': 0.03, 'max': 0.20, 'type': float}, 
    'th_sell':         {'min': 0.03, 'max': 0.20, 'type': float}, 
    'temp_T':          {'min': 1.1,  'max': 2.5,  'type': float}, 
    
    # 리스크 관리 (7개)
    'atr_multiplier':  {'min': 1.0,  'max': 4.0,  'type': float}, # [수정] 초기 SL 승수
    'tp_ratio':        {'min': 1.5,  'max': 8.0,  'type': float}, # [수정] 트레일링 SL 승수
    'min_hold':        {'min': 1,    'max': 10,    'type': int},
    'cooldown':        {'min': 0,    'max': 3,    'type': int},
    'risk_per_trade':  {'min': 0.01,'max': 0.10, 'type': float},

}
N_RESERVOIR_FIXED = 400

try:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
except RuntimeError:
    pass

def _rand_in(key):
    spec = PARAM_RANGES[key]
    if spec['type'] is int:
        return random.randint(spec['min'], spec['max'])
    return random.uniform(spec['min'], spec['max'])

def generate_individual():
    """ 10개 파라미터 개체 생성 """
    return [
        _rand_in('spectral_radius'),
        _rand_in('sparsity'),
        _rand_in('th_buy'),
        _rand_in('th_sell'),
        _rand_in('temp_T'),
        _rand_in('atr_multiplier'),
        _rand_in('tp_ratio'),
        _rand_in('min_hold'),
        _rand_in('cooldown'),
        _rand_in('risk_per_trade'),
        # [삭제됨] commission, slippage, enforce_delay
    ]

# =========================
# 피트니스 함수 (고정값 적용)
# =========================
def fitness_function_with_backtesting(params, train_df: pd.DataFrame, test_df: pd.DataFrame, Technical_Signals=None):
    # [수정] 10개 파라미터만 언패킹
    (
        spectral_radius, sparsity, 
        th_buy, th_sell, temp_T,
        atr_multiplier, tp_ratio,
        min_hold, cooldown, risk_per_trade
    ) = params

    # [추가] 고정 파라미터
    n_reservoir = N_RESERVOIR_FIXED
    commission_fixed = 0.0005
    slippage_fixed = 0.0003
    enforce_delay = 1

    # 경계 보정 (Clamping)
    def clamp(v, key):
        spec = PARAM_RANGES[key]
        if spec['type'] is int:
            return int(max(spec['min'], min(int(round(v)), spec['max'])))
        return float(max(spec['min'], min(float(v), spec['max'])))

    spectral_radius = clamp(spectral_radius, 'spectral_radius')
    sparsity        = clamp(sparsity,        'sparsity')
    th_buy          = clamp(th_buy,          'th_buy')
    th_sell         = clamp(th_sell,         'th_sell')
    temp_T          = clamp(temp_T,          'temp_T')
    atr_multiplier  = clamp(atr_multiplier,  'atr_multiplier')
    tp_ratio        = clamp(tp_ratio,        'tp_ratio')
    min_hold        = clamp(min_hold,        'min_hold')
    cooldown        = clamp(cooldown,        'cooldown')
    risk_per_trade  = clamp(risk_per_trade,  'risk_per_trade')

    # 제약/가중치
    MDD_HARD_CAP = 35.0
    MIN_TRADES   = 7
    MDD_WEIGHT   = 0.04
    LOG_RETURN_W = 0.15

    try:
        # 1) ESN 신호 생성
        sig_df = esn_signals(
            train_df=train_df,
            test_df=test_df,
            Technical_Signals=Technical_Signals,
            n_reservoir=n_reservoir, # 고정값
            spectral_radius=spectral_radius,
            sparsity=sparsity,
            th_buy=th_buy,
            th_sell=th_sell,
            temp_T=temp_T,
            # random_state는 GA 실행 시 np.random.seed로 제어됨
        )
        if sig_df.empty or 'Predicted_Signals' not in sig_df.columns:
            return (-1000.0,)

        # 2) 백테스트 데이터 구성 (1봉 지연 적용)
        bt_df = test_df.copy()
        sig = sig_df['Predicted_Signals'].reindex(bt_df.index).fillna(0).astype(np.int8)
        if enforce_delay == 1:
            sig = sig.shift(1).fillna(0).astype(np.int8)  
        bt_df['Predicted_Signals'] = sig

        # 3) 백테스트 실행 (고정 수수료 사용)
        bt = Backtest(
            bt_df, PredictedSignalStrategy,
            cash=10000,
            commission=commission_fixed, # 고정값
             slippage=slippage_fixed,   # 필요 시 주석 해제
            exclusive_orders=True
        )
        stats = bt.run(
            atr_multiplier=float(atr_multiplier),
            tp_ratio=float(tp_ratio),
            min_hold=int(min_hold),
            cooldown=int(cooldown),
            risk_per_trade=float(risk_per_trade)
        )

        # 4) 기본 제약 검사
        n_trades = int(stats['# Trades'])
        if n_trades < MIN_TRADES: return (-500.0,)

        sharpe = float(stats.get('Sharpe Ratio', np.nan))
        if not np.isfinite(sharpe): return (-500.0,)

        mdd = abs(float(stats['Max. Drawdown [%]']))
        if mdd >= MDD_HARD_CAP: return (-800.0,)

        total_return = float(stats['Return [%]'])
        cagr = float(stats.get('Return (Ann.) [%]', 0.0)) / 100.0

        # 5) 최종 피트니스 계산 (Calmar Ratio 중심)
        calmar = float(stats.get('Calmar Ratio', 0.0))
        
        if calmar > 0 and mdd > 1.0:
            fitness = calmar
        else:
            fitness = -100.0 + total_return # 패널티 완화

        # 보정
        fitness *= (0.9 + 0.1 * min(1.0, n_trades / (MIN_TRADES * 2)))
        if np.isfinite(cagr):
            fitness *= (0.8 + 0.2 * max(0.0, min(1.0, cagr / 0.15)))

        if not np.isfinite(fitness): return (-500.0,)
        return (float(np.clip(fitness, -1000.0, 1000.0)),)

    except Exception:
        return (-1000.0,)

# =========================
# DEAP 설정
# =========================
toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, generate_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", fitness_function_with_backtesting)
toolbox.register("mate", tools.cxBlend, alpha=0.5)

# 10차원 뮤테이션 시그마 (수수료/슬리피지 제외)
sigma_vals = [
    (PARAM_RANGES['spectral_radius']['max'] - PARAM_RANGES['spectral_radius']['min']) * 0.1,
    (PARAM_RANGES['sparsity']['max']        - PARAM_RANGES['sparsity']['min'])        * 0.1,
    (PARAM_RANGES['th_buy']['max']          - PARAM_RANGES['th_buy']['min'])          * 0.1,
    (PARAM_RANGES['th_sell']['max']         - PARAM_RANGES['th_sell']['min'])         * 0.1,
    (PARAM_RANGES['temp_T']['max']          - PARAM_RANGES['temp_T']['min'])          * 0.1,
    (PARAM_RANGES['atr_multiplier']['max']  - PARAM_RANGES['atr_multiplier']['min'])  * 0.1,
    (PARAM_RANGES['tp_ratio']['max']        - PARAM_RANGES['tp_ratio']['min'])        * 0.1,
    (PARAM_RANGES['min_hold']['max']        - PARAM_RANGES['min_hold']['min'])        * 0.1,
    (PARAM_RANGES['cooldown']['max']        - PARAM_RANGES['cooldown']['min'])        * 0.1,
    (PARAM_RANGES['risk_per_trade']['max']  - PARAM_RANGES['risk_per_trade']['min'])  * 0.1,
]
toolbox.register("mutate", tools.mutGaussian, mu=[0]*len(sigma_vals), sigma=sigma_vals, indpb=0.12)
toolbox.register("select", tools.selTournament, tournsize=4)

# =========================
# GA 실행 유틸
# =========================
def run_genetic_algorithm(train_df_ga, test_df_ga, technical_signals_list,
                          pop_size=50, num_generations=20, cxpb=0.7, mutpb=0.4, random_seed=42):
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    toolbox.evaluate.keywords['train_df'] = train_df_ga
    toolbox.evaluate.keywords['test_df'] = test_df_ga
    toolbox.evaluate.keywords['Technical_Signals'] = technical_signals_list

    population = toolbox.population(n=pop_size)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean); stats.register("std", np.std)
    stats.register("min", np.min);  stats.register("max", np.max)
    hof = tools.HallOfFame(1)

    population, log = algorithms.eaSimple(
        population, toolbox, cxpb, mutpb, num_generations,
        stats=stats, halloffame=hof, verbose=True
    )
    best_individual = hof[0]
    return best_individual, log

# =========================
# CV → 후보 → 교차평가 → 최종
# =========================
def find_best_params_with_cv(df_for_tuning, technical_signals_list,
                             n_splits=5, pop_size=30, num_generations=15, random_seed=42):
    print(f"--- {n_splits}-분할 교차 검증으로 파라미터 최적화 시작 ---")
    tscv = TimeSeriesSplit(n_splits=n_splits)
    candidate_params_list = []
    all_splits = list(tscv.split(df_for_tuning))

    for i, (tr_idx, va_idx) in enumerate(all_splits):
        print(f"\n--- CV Fold {i+1}/{n_splits} (후보 파라미터 탐색) ---")
        train_cv_df = df_for_tuning.iloc[tr_idx]
        val_cv_df   = df_for_tuning.iloc[va_idx]
        best_params_for_fold, _ = run_genetic_algorithm(
            train_df_ga=train_cv_df, test_df_ga=val_cv_df,
            technical_signals_list=technical_signals_list,
            pop_size=pop_size, num_generations=num_generations,
            random_seed=random_seed + i
        )
        candidate_params_list.append(best_params_for_fold)
        print(f"Fold {i+1} 최적 후보: {best_params_for_fold}")

    print("\n" + "="*50)
    print("후보 파라미터 교차 평가 시작...")
    avg_scores = []
    for i, params in enumerate(candidate_params_list):
        scores_for_params = []
        print(f"  -> 후보 {i+1} 평가 중...")
        for j, (tr_idx, va_idx) in enumerate(all_splits):
            train_cv_df = df_for_tuning.iloc[tr_idx]
            val_cv_df   = df_for_tuning.iloc[va_idx]
            fitness = fitness_function_with_backtesting(
                params, train_cv_df, val_cv_df, technical_signals_list
            )[0]
            scores_for_params.append(fitness)
        avg_score = float(np.mean(scores_for_params))
        avg_scores.append(avg_score)
        print(f"  후보 {i+1}의 {n_splits}개 Fold 평균 점수: {avg_score:.4f}")

    best_idx = int(np.argmax(avg_scores))
    final_best_params = candidate_params_list[best_idx]
    print("\n" + "="*50)
    print("교차 검증 기반 파라미터 최적화 완료!")
    print(f"최종 선택된 최적 파라미터: {final_best_params}")
    print(f"(후보 {best_idx+1} 평균 점수 {avg_scores[best_idx]:.4f})")
    print("="*50 + "\n")
    return final_best_params

# =========================
# 롤링 포워드 검증
# =========================
def rolling_forward_split(df: pd.DataFrame, n_splits: int, initial_train_ratio: float = 0.5):
    total_len = len(df)
    initial_train_size = int(total_len * initial_train_ratio)
    remaining_len = total_len - initial_train_size
    val_size = 0 if n_splits == 0 else remaining_len // n_splits
    for i in range(n_splits):
        train_end = initial_train_size + i * val_size
        val_end   = train_end + val_size
        train_df = df.iloc[:train_end].copy()
        val_df   = df.iloc[train_end:val_end].copy()
        if val_df.empty:
            continue
        yield train_df, val_df

def perform_final_backtest(train_df, test_df, best_params, technical_signals_list,
                           random_state=42, fold_num=None):
    
    # 1) 10개 파라미터 언패킹
    (spectral_radius, sparsity, th_buy, th_sell, temp_T,
     atr_multiplier, tp_ratio, 
     min_hold, cooldown, risk_per_trade) = best_params

    # 2) clamp 함수 추가 (fitness_function과 동일)
    def clamp(v, key):
        spec = PARAM_RANGES[key]
        if spec['type'] is int:
            return int(max(spec['min'], min(int(round(v)), spec['max'])))
        return float(max(spec['min'], min(float(v), spec['max'])))

    # 3) 모든 파라미터 보정
    spectral_radius = clamp(spectral_radius, 'spectral_radius')
    sparsity        = clamp(sparsity,        'sparsity')
    th_buy          = clamp(th_buy,          'th_buy')
    th_sell         = clamp(th_sell,         'th_sell')
    temp_T          = clamp(temp_T,          'temp_T')
    atr_multiplier  = clamp(atr_multiplier,  'atr_multiplier')
    tp_ratio        = clamp(tp_ratio,        'tp_ratio')
    min_hold        = clamp(min_hold,        'min_hold')
    cooldown        = clamp(cooldown,        'cooldown')
    risk_per_trade  = clamp(risk_per_trade,  'risk_per_trade')

    # 4) 이하 원래 코드 동일
    n_reservoir = N_RESERVOIR_FIXED
    commission_fixed = 0.0005
    slippage_fixed = 0.0003
    enforce_delay = 1

    sig_df = esn_signals(
        train_df=train_df, test_df=test_df,
        Technical_Signals=technical_signals_list,
        n_reservoir=int(n_reservoir), spectral_radius=float(spectral_radius),
        sparsity=float(sparsity),
        th_buy=float(th_buy), th_sell=float(th_sell), temp_T=float(temp_T),
    )
    if not isinstance(sig_df, pd.DataFrame) or sig_df.empty or 'Predicted_Signals' not in sig_df.columns:
        print("ESN 모델에서 유효한 신호가 생성되지 않았습니다. 백테스트 건너뜀.")
        return None, None

    bt_df = test_df.copy()  # 추천: TA 필터 그대로 사용
    sig = sig_df['Predicted_Signals'].reindex(bt_df.index).fillna(0).astype(np.int8)
    if int(enforce_delay) == 1:
        sig = sig.shift(1).fillna(0).astype(np.int8)
    bt_df['Predicted_Signals'] = sig

    bt_final = Backtest(
        bt_df, PredictedSignalStrategy,
        cash=10000,
        commission=commission_fixed,
        slippage=slippage_fixed,
        exclusive_orders=True
    )

    stats_final = bt_final.run(
        atr_multiplier=float(atr_multiplier), 
        tp_ratio=float(tp_ratio),           
        min_hold=int(min_hold),
        cooldown=int(cooldown),
        risk_per_trade=float(risk_per_trade)
    )

    print("\n백테스팅 결과:")
    print(stats_final)
    
    if fold_num == "last":
        bt_final.plot(filename='final_fold_backtest_results', open_browser=False)

    return stats_final, sig_df

def esn_rolling_forward(df, technical_signals_list,
                        n_splits_cv=5, n_splits_forward=3,
                        pop_size=30, num_generations=15,
                        random_seed=42, initial_train_ratio=0.7):
    train_end_idx = int(len(df) * initial_train_ratio)
    df_for_tuning = df.iloc[:train_end_idx]
    best_params = find_best_params_with_cv(
        df_for_tuning=df_for_tuning,
        technical_signals_list=technical_signals_list,
        n_splits=n_splits_cv, pop_size=pop_size,
        num_generations=num_generations, random_seed=random_seed
    )

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
                technical_signals_list=technical_signals_list,
                fold_num=("last" if i == n_splits_forward - 1 else None),
                random_state=random_seed
            )
            if stats is not None:
                all_fold_stats.append(stats)
                if i == n_splits_forward - 1:
                    last_fold_signals = signals
        except Exception as e:
            print(f"폴드 {i+1} 백테스팅 중 오류 발생: {e}")
            traceback.print_exc()

    print("\n" + "="*50)
    print("롤링 포워드 교차 검증 최종 결과 요약:")
    print("="*50)
    return best_params, all_fold_stats, last_fold_signals