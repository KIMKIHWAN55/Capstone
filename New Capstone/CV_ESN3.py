# CV_ESN3.py (전체 교체용 코드)
#
# 1. 'macd_params' NameError 수정 (ts_optimization 함수 내)
# 2. 'random_state' NameError 수정 (run_genetic_algorithm -> fitness_function -> esn_signals로 전달되도록 수정)
# 3. [수정] PredictedSignalStrategy를 'ATR 트레일링 스톱' 버전으로 교체
# 4. [유지] ts_optimization (TA 지표 GA 최적화) 로직은 그대로 MANTENIDO.

import pandas as pd
import numpy as np
import random
from deap import base, creator, tools, algorithms
from backtesting import Backtest, Strategy
from ESN_Signals import esn_signals # ESN_Signals.py (수정본) 필요
import warnings
import multiprocessing
import talib 
import traceback # 오류 추적용

# --- 1. 사용할 TA 지표 모듈 임포트 ---
import CPM2 as cpm
import RSI2 as rsi
import MACD as macd
import BollingerBands2 as bb 
# import importlib  # 사용 안함 → 제거 권장

warnings.filterwarnings('ignore')

# ==================================================
# [수정] 2. 백테스팅 전략 클래스 (트레일링 스톱 버전으로 수정)
#    (ATR 기반 초기 손절, ATR 기반 트레일링 스톱 적용)
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
        self.sma_trend = self.I(talib.SMA, self.data.Close, timeperiod=20, name="SMA20")

        
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
        should_enter = (sig == -1) 

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

# ==================================================
# [유지] 3. ESN + 리스크 하이퍼파라미터 최적화 (GA)
# (사용자 버전 - 12개 파라미터)
# ==================================================

# ESN 파라미터(5개) + 리스크 파라미터(7개) = 총 12개
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
N_RESERVOIR_FIXED = 400 # ESN 리저버 크기는 고정

def _rand_in(key):
    """ PARAM_RANGES에서 랜덤 값 생성 헬퍼 """
    spec = PARAM_RANGES[key]
    if spec['type'] is int:
        return random.randint(spec['min'], spec['max'])
    return random.uniform(spec['min'], spec['max'])

def generate_individual():
    """ 12개 파라미터 개체 생성 """
    return [
        _rand_in('spectral_radius'),
        _rand_in('sparsity'),
        _rand_in('th_buy'),
        _rand_in('th_sell'),
        _rand_in('temp_T'),
        _rand_in('atr_multiplier'),
        _rand_in('tp_ratio'), # [수정] 트레일링 승수
        _rand_in('min_hold'),
        _rand_in('cooldown'),
        _rand_in('risk_per_trade'),

    ]

### 수정 2-2: random_state 인자를 받도록 함수 시그니처 수정 ###
def fitness_function_with_backtesting(params, train_df: pd.DataFrame, test_df: pd.DataFrame, 
                                      Technical_Signals=None, random_state: int = 42):
    """ (수정) ESN_GA.py의 정교한 적합도 함수로 교체 """
    
    # 12개 파라미터 언패킹
    (
        spectral_radius, sparsity, th_buy, th_sell, temp_T,
        atr_multiplier, tp_ratio, # [수정] tp_ratio는 트레일링 승수
        min_hold, cooldown, risk_per_trade,

    ) = params
    # [추가] 고정된 수수료/슬리피지 설정
    commission_fixed = 0.0005
    slippage_fixed = 0.0003
    enforce_delay = 1
    
    # 동일봉 체결 방지 (1봉 지연) 하드코딩
    enforce_delay = 1 

    # --- 파라미터 경계 보정 (Clamping) ---
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
    tp_ratio        = clamp(tp_ratio,        'tp_ratio') # [수정] 트레일링 승수
    min_hold        = clamp(min_hold,        'min_hold')
    cooldown        = clamp(cooldown,        'cooldown')
    risk_per_trade  = clamp(risk_per_trade,  'risk_per_trade')

    # --- ESN_GA.py의 적합도 계산 로직 ---
    MDD_HARD_CAP = 35.0   # MDD 35% 초과 시 패널티
    MIN_TRADES   = 7    # 최소 거래 25회 미만 시 패널티 (사용자 설정)
    MDD_WEIGHT   = 0.04    # Sharpe 대비 MDD 패널티 가중치
    LOG_RETURN_W = 0.15    # 총수익 보너스 가중치

    try:
        # 1) ESN 신호 생성 (수정된 ESN_Signals.py 호출)
        backtest_signals_df = esn_signals(
            train_df=train_df,
            test_df=test_df,
            Technical_Signals=Technical_Signals,
            n_reservoir=N_RESERVOIR_FIXED, 
            spectral_radius=spectral_radius,
            sparsity=sparsity,
            th_buy=th_buy,
            th_sell=th_sell,
            temp_T=temp_T,
            random_state=random_state  ### 수정 2-3: random_state 전달 ###
        )
        if backtest_signals_df.empty or 'Predicted_Signals' not in backtest_signals_df.columns:
            return (-1000.0,) 
        
        # 2) 백테스트 데이터 구성
        backtest_data = test_df.copy()
        sig = backtest_signals_df['Predicted_Signals'].reindex(backtest_data.index).fillna(0).astype(np.int8)
        
        if enforce_delay == 1:
            sig = sig.shift(1).fillna(0).astype(np.int8)
            
        backtest_data['Predicted_Signals'] = sig
        
        # 3) 백테스트 실행 (수정된 전략 사용)
        bt = Backtest(backtest_data, PredictedSignalStrategy,
                      cash=10000, 
                      commission=commission_fixed, 
                      #slippage=float(slippage),
                      exclusive_orders=True) 
        
        stats = bt.run(
            atr_multiplier=float(atr_multiplier), # [수정] 초기 SL
            tp_ratio=float(tp_ratio),           # [수정] 트레일링 SL
            min_hold=int(min_hold),
            cooldown=int(cooldown),
            risk_per_trade=float(risk_per_trade)
        )
        
        # 4) 기본 제약 (ESN_GA.py 로직)
        n_trades = int(stats['# Trades'])
        if n_trades < MIN_TRADES:
            return (-500.0,)

        sharpe = float(stats.get('Sharpe Ratio', np.nan))
        if not np.isfinite(sharpe):
            return (-500.0,)

        mdd = abs(float(stats['Max. Drawdown [%]']))
        if mdd >= MDD_HARD_CAP:
            return (-800.0,)

        total_return = float(stats['Return [%]'])
        cagr = float(stats.get('Return (Ann.) [%]', 0.0)) / 100.0

        # 5) 최종 피트니스 (ESN_GA.py 로직)
        log_ret_bonus = np.log1p(max(0.0, total_return)) * LOG_RETURN_W if total_return > -100 else 0.0
        fitness = sharpe - (MDD_WEIGHT * mdd) + log_ret_bonus

        # 거래수·CAGR 보정(소폭)
        fitness *= (0.9 + 0.1 * min(1.0, n_trades / (MIN_TRADES * 2)))
        if np.isfinite(cagr):
            fitness *= (0.8 + 0.2 * max(0.0, min(1.0, cagr / 0.15)))

        if not np.isfinite(fitness):
            return (-500.0,)
        return (float(np.clip(fitness, -1000.0, 1000.0)),)

    except Exception as e:
        print(f"!!! 피트니스 함수 오류 발생: {e}")
        traceback.print_exc() # <--- 오류의 상세 내용을 출력합니다.
        return (-1000.0,)

def init_deap_creator():
    # (FitnessMax 사용 (Sharpe 기반))
    for name in ["FitnessMax", "Individual"]:
        if hasattr(creator, name):
            delattr(creator, name)
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

def run_genetic_algorithm(train_df_ga: pd.DataFrame, test_df_ga: pd.DataFrame, technical_signals_list: list,
                          pop_size: int = 50, num_generations: int = 20, cxpb: float = 0.7, mutpb: float = 0.2,
                          random_seed: int = 42):
    random.seed(random_seed)
    np.random.seed(random_seed)

    init_deap_creator() 

    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual, generate_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    ### 수정 2-1: GA의 random_seed를 'random_state' 인자로 전달 ###
    toolbox.register("evaluate", fitness_function_with_backtesting,
                     train_df=train_df_ga,
                     test_df=test_df_ga,
                     Technical_Signals=technical_signals_list,
                     random_state=random_seed) # ESN 시드 고정을 위해 전달

    toolbox.register("mate", tools.cxBlend, alpha=0.5)

    # 12개 파라미터에 대한 뮤테이션 시그마
    sigma_vals = [
        (PARAM_RANGES['spectral_radius']['max'] - PARAM_RANGES['spectral_radius']['min']) * 0.1,
        (PARAM_RANGES['sparsity']['max']        - PARAM_RANGES['sparsity']['min'])        * 0.1,
        (PARAM_RANGES['th_buy']['max']          - PARAM_RANGES['th_buy']['min'])          * 0.1,
        (PARAM_RANGES['th_sell']['max']         - PARAM_RANGES['th_sell']['min'])         * 0.1,
        (PARAM_RANGES['temp_T']['max']          - PARAM_RANGES['temp_T']['min'])          * 0.1,
        (PARAM_RANGES['atr_multiplier']['max']  - PARAM_RANGES['atr_multiplier']['min'])  * 0.1,
        (PARAM_RANGES['tp_ratio']['max']        - PARAM_RANGES['tp_ratio']['min'])        * 0.1, # [수정] 트레일링 승수
        (PARAM_RANGES['min_hold']['max']        - PARAM_RANGES['min_hold']['min'])        * 0.1,
        (PARAM_RANGES['cooldown']['max']        - PARAM_RANGES['cooldown']['min'])        * 0.1,
        (PARAM_RANGES['risk_per_trade']['max']  - PARAM_RANGES['risk_per_trade']['min'])  * 0.1,
    ]
    toolbox.register("mutate", tools.mutGaussian,
                     mu=[0]*len(sigma_vals),
                     sigma=sigma_vals,
                     indpb=0.12) 
    
    toolbox.register("select", tools.selTournament, tournsize=4) 

    population = toolbox.population(n=pop_size)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    hof = tools.HallOfFame(1)
    
# --- 멀티프로세싱 (디버깅을 위해 강제 비활성화) ---
    print("!!! 디버깅 모드: 멀티프로세싱을 비활성화하고 싱글 코어로 실행합니다. !!!")
    toolbox.register("map", map)
# --- 멀티프로세싱 ---
    try:
        pool = multiprocessing.Pool()
        toolbox.register("map", pool.map)
    except Exception as e:
        print(f"멀티프로세싱 풀 생성 실패: {e}. 싱글 코어로 실행합니다.")
        toolbox.register("map", map)
    
    population, log = algorithms.eaSimple(population, toolbox, cxpb, mutpb, num_generations,
                                          stats=stats, halloffame=hof, verbose=True)
    
    if 'pool' in locals():
        pool.close()
        pool.join()

    best_individual = hof[0]
    print(f"\nGA 최적화 완료 - 최적 하이퍼파라미터 (12개): {best_individual}")
    print(f"GA 최적화 완료 - 최고 Fitness (Sharpe기반): {best_individual.fitness.values[0]:.4f}")

    return best_individual, log
    
def perform_final_backtest(train_df: pd.DataFrame, test_df: pd.DataFrame, best_params: list, technical_signals_list: list,
                           random_state: int = 42, commission: float = 0.0005, slippage: float = 0.0003):
    
    """ (수정) ESN_GA.py의 정교한 백테스트 함수로 교체 """
    
    # 12개 파라미터 언패킹
    (
        spectral_radius, sparsity, th_buy, th_sell, temp_T,
        atr_multiplier, tp_ratio, # [수정] tp_ratio는 트레일링 승수
        min_hold, cooldown, risk_per_trade,

    ) = best_params
    
    # 동일봉 체결 방지 (1봉 지연) 하드코딩
    enforce_delay = 1

    # --- 파라미터 경계 보정 (Clamping) ---
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
    tp_ratio        = clamp(tp_ratio,        'tp_ratio') # [수정] 트레일링 승수
    min_hold        = clamp(min_hold,        'min_hold')
    cooldown        = clamp(cooldown,        'cooldown')
    risk_per_trade  = clamp(risk_per_trade,  'risk_per_trade')



    print(f"\n--- 최적화된 파라미터로 최종 ESN 학습 및 백테스팅 ---")
    print(f"  (ESN) n_reservoir: {N_RESERVOIR_FIXED}")
    print(f"  (ESN) spectral_radius: {spectral_radius:.4f}, sparsity: {sparsity:.4f}")
    print(f"  (ESN) th_buy: {th_buy:.4f}, th_sell: {th_sell:.4f}, temp_T: {temp_T:.4f}")
    print(f"  (Risk) atr_mult (초기 SL): {atr_multiplier:.2f}, tp_ratio (트레일링): {tp_ratio:.2f}")
    print(f"  (Risk) min_hold: {min_hold}, cooldown: {cooldown}")
    print(f"  (Risk) risk_per_trade: {risk_per_trade:.4f}, commission: {commission:.4f}, slippage: {slippage:.4f}")
    print(f"  (Test) enforce_delay: {enforce_delay} (1봉 지연 적용)")


    final_backtest_signals_df = esn_signals(
        train_df=train_df,
        test_df=test_df,
        Technical_Signals=technical_signals_list,
        n_reservoir=N_RESERVOIR_FIXED,
        spectral_radius=spectral_radius,
        sparsity=sparsity,
        th_buy=th_buy,
        th_sell=th_sell,
        temp_T=temp_T,
        random_state=random_state # (여기는 원래 random_state를 받고 있었음)
    )

    if not isinstance(final_backtest_signals_df, pd.DataFrame) or final_backtest_signals_df.empty or 'Predicted_Signals' not in final_backtest_signals_df.columns:
        print("최종 ESN 모델에서 유효한 신호가 생성되지 않았습니다. 백테스팅을 건너뜀.")
        return None, None

    final_backtest_data = test_df.copy()
    sig = final_backtest_signals_df['Predicted_Signals'].reindex(final_backtest_data.index).fillna(0).astype(np.int8)
    
    if int(enforce_delay) == 1:
        sig = sig.shift(1).fillna(0).astype(np.int8)
        
    final_backtest_data['Predicted_Signals'] = sig

    bt_final = Backtest(final_backtest_data, PredictedSignalStrategy,
                        cash=10000, 
                        commission=float(commission),
                        #slippage=float(slippage),
                        exclusive_orders=True)
    
    stats_final = bt_final.run(
        atr_multiplier=float(atr_multiplier), # [수정] 초기 SL
        tp_ratio=float(tp_ratio),           # [수정] 트레일링 SL
        min_hold=int(min_hold),
        cooldown=int(cooldown),
        risk_per_trade=float(risk_per_trade)
    )

    print("\n최종 백테스팅 결과 (최적화된 파라미터):")
    print(stats_final)
    
    return stats_final, final_backtest_signals_df

# ==================================================
# [유지] (이하 롤링 포워드 및 TA 최적화 로직은 사용자 버전 그대로)
# ==================================================

def rolling_forward_split_3way(df: pd.DataFrame, n_splits: int, initial_train_ratio: float = 0.5):
    total_len = len(df)
    initial_train_and_val_size = int(total_len * initial_train_ratio)
    remaining_len = total_len - initial_train_and_val_size
    
    if n_splits <= 0: return
    test_size = remaining_len // n_splits
    if test_size == 0 and remaining_len > 0:
        test_size = remaining_len
        n_splits = 1
    elif test_size == 0:
        return

    val_size = test_size
    initial_train_size = initial_train_and_val_size - val_size
    
    if initial_train_size <= 0: return

    print(f"--- 3-Way 분할 설정 ---")
    print(f"초기 Train 크기: {initial_train_size}, Validation 크기: {val_size}, Test 크기: {test_size}")
    print(f"총 {n_splits}개 폴드 생성")
    print(f"----------------------")

    for i in range(n_splits):
        train_end_idx = initial_train_size + i * test_size
        train_df = df.iloc[:train_end_idx].copy()
        
        val_start_idx = train_end_idx
        val_end_idx = val_start_idx + val_size
        val_df = df.iloc[val_start_idx:val_end_idx].copy()
        
        test_start_idx = val_end_idx
        test_end_idx = test_start_idx + test_size
        
        if i == n_splits - 1:
            test_end_idx = total_len
            
        test_df = df.iloc[test_start_idx:test_end_idx].copy()

        if val_df.empty or test_df.empty:
            continue
            
        yield train_df, val_df, test_df

# 4. TA 지표 "원시 값" 피처 생성 함수
def generate_signals(train_df, val_df, test_df, bb_params, rsi_params, macd_params):
    print("    - (Raw) BB/RSI/MACD 피처 생성 중...")
    train_df_sig = train_df.copy()
    val_df_sig = val_df.copy()   
    test_df_sig = test_df.copy()
    all_dfs = [train_df_sig, val_df_sig, test_df_sig] 

    # --- 0. (수정) CPM 타겟 컬럼 보존 ---
    if 'cpm_point_type' in train_df.columns:
        train_df_sig['cpm_point_type'] = train_df['cpm_point_type']
    if 'cpm_point_type' in val_df.columns:
        val_df_sig['cpm_point_type'] = val_df['cpm_point_type']
    if 'cpm_point_type' in test_df.columns:
        test_df_sig['cpm_point_type'] = test_df['cpm_point_type']

    # --- 1. BB 피처 (%B 값) ---
    try:
        period, std_up, std_low = bb_params
        period = int(round(period))
        for df in all_dfs:
            upper, middle, lower = talib.BBANDS(df['Close'], timeperiod=period, nbdevup=std_up, nbdevdn=std_low)
            df['BB_Value'] = (df['Close'] - lower) / (upper - lower)
            # (수정 6) NaN/inf 발생 시 0.5(중립)로 채움
            df['BB_Value'] = df['BB_Value'].fillna(0.5) 
            df.loc[~np.isfinite(df['BB_Value']), 'BB_Value'] = 0.5 
    except Exception as e:
        print(f"    - ⚠️ BB 피처 생성 오류: {e}. 0.5로 채움.")
        for df in all_dfs: df['BB_Value'] = 0.5

    # --- 2. RSI 피처 (Raw Value) ---
    try:
        x, ob, os, p, q = rsi_params
        x = int(round(x))
        for df in all_dfs:
            df['RSI_Value'] = talib.RSI(df['Close'], timeperiod=x)
            df['RSI_Value'] = df['RSI_Value'].fillna(50) # 50 (중립)
    except Exception as e:
        print(f"    - ⚠️ RSI 피처 생성 오류: {e}. 50으로 채움.")
        for df in all_dfs: df['RSI_Value'] = 50

    # --- 3. MACD 피처 (Histogram 값) ---
    try:
        fast, slow, signal, buy_th, sell_th = macd_params
        fast, slow, signal = int(round(fast)), int(round(slow)), int(round(signal))
        for df in all_dfs:
            _, _, hist = talib.MACD(df['Close'], fastperiod=fast, slowperiod=slow, signalperiod=signal)
            df['MACD_Value'] = hist
            df['MACD_Value'] = df['MACD_Value'].fillna(0) # 0 (중립)
    except Exception as e:
        print(f"    - ⚠️ MACD 피처 생성 오류: {e}. 0으로 채움.")
        for df in all_dfs: df['MACD_Value'] = 0

    # --- 4. [수정 6] 추세 필터 피처 (SMA 200) ---
    try:
        sma_period = 200
        for df in all_dfs:
            # (수정 6) 데이터가 200일 미만이면 편향을 막기 위해 0으로 채움
            if len(df) < sma_period:
                df['Trend_Filter'] = 0
            else:
                sma200 = talib.SMA(df['Close'], timeperiod=sma_period)
                df['Trend_Filter'] = 1 
                df.loc[df['Close'] < sma200, 'Trend_Filter'] = -1
                # (수정 6) NaN 값을 1(상승)이 아닌 0(중립)으로 채움
                df['Trend_Filter'] = df['Trend_Filter'].fillna(0) 
    except Exception as e:
        print(f"    - ⚠️ 추세 필터 생성 오류: {e}. 0으로 채움.")
        for df in all_dfs: df['Trend_Filter'] = 0

    
    return train_df_sig, val_df_sig, test_df_sig
    
# [유지] 5. TA 지표 파라미터 최적화 (CPM 피팅)
def ts_optimization(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    1. 각 기술적 지표의 최적 파라미터를 각각 독립적으로 찾습니다. (CPM 피팅, Train 데이터 기준)
    2. 모든 최적 파라미터를 구한 후, "원시 피처(Raw Feature)" 컬럼들을 추가합니다.
    """
    
    # --- 1단계: 각 기술적 지표의 최적 파라미터를 찾기 (Train 데이터 기준) ---
    
    print("    - 볼린저 밴드(BB) 파라미터 최적화...")
    bb_best_params, _, _ = bb.run_BB_ga_optimization(train_df, generations=50, population_size=50)
    print(f"    - BB 최적 파라미터: {bb_best_params}")

    print("    - RSI 파라미터 최적화...")
    rsi_best_params, _, _ = rsi.run_RSI_ga_optimization(train_df, generations=50, population_size=50)
    print(f"    - RSI 최적 파라미터: {rsi_best_params}")

    print("    - MACD 파라미터 최적화...")
    macd_best_params, _, _ = macd.run_MACD_ga_optimization(train_df, generations=50, population_size=50)
    print(f"    - MACD 최적 파라미터: {macd_best_params}")

    # --- 2단계: "원시 피처" 컬럼 일괄 추가 ---
    print("\n    - 모든 최적 파라미터로 [원시 피처] 컬럼 생성 중...")

    train_df_with_signals, val_df_with_signals, test_df_with_signals = generate_signals(
        train_df.copy(), val_df.copy(), test_df.copy(),
        bb_best_params, 
        rsi_best_params, 
        macd_best_params  # macd_params -> macd_best_params  (이미 수정되어 있음)
    )

    # (수정) ESN에 입력할 "피처 이름 리스트" (Trend_Filter 추가!)
    technical_signals_list = ['BB_Value', 'RSI_Value', 'MACD_Value', 'Trend_Filter']

    print(f"    - ESN에 사용할 피처: {technical_signals_list} (+ Close_p)")
    
    return train_df_with_signals, val_df_with_signals, test_df_with_signals, technical_signals_list

# 6. 메인 롤링 포워드 실행 함수
def esn_rolling_forward_safe(df: pd.DataFrame, n_splits: int = 5, initial_train_ratio: float = 0.7,
                             pop_size: int = 50, num_generations: int = 50, 
                             random_seed: int = 42,
                             commission: float = 0.0005, slippage: float = 0.0003):
    
    total_returns = []
    bh_returns = []
    total_mdd = []
    total_sharpe = [] 
    best_params_per_fold = []
    
    # 3-Way 분할 (Train / Validation / Test)
    splits = list(rolling_forward_split_3way(df, n_splits, initial_train_ratio))
    if not splits:
        print("유효한 데이터 분할이 생성되지 않았습니다.")
        return [], [], None

    print(f"\n--- 롤링 포워드 교차 검증 (T/V/T 분리) 시작 ---")
    
    for i, (train_df, validation_df, test_df) in enumerate(splits):
        print("\n" + "="*50)
        print(f"--- 폴드 {i+1} / {n_splits} ---")
        print(f"Train: {train_df.index.min()} ~ {train_df.index.max()} ({len(train_df)}일)")
        print(f"Valid: {validation_df.index.min()} ~ {validation_df.index.max()} ({len(validation_df)}일)")
        print(f"Test:  {test_df.index.min()} ~ {test_df.index.max()} ({len(test_df)}일)")
        print("="*50)
        
        try:
            # --- 1단계: 각 구간별 CPM 정답 생성 ---
            print(f"[{i+1}/{n_splits}] 1단계: 각 구간별 CPM 정답 생성 중...")
            _, train_df_with_cpm = cpm.cpm_model(train_df, column='Close', P=0.05, T=5)
            _, val_df_with_cpm = cpm.cpm_model(validation_df, column='Close', P=0.05, T=5)
            _, test_df_with_cpm = cpm.cpm_model(test_df, column='Close', P=0.05, T=5)
        
            # --- 2단계: TA 최적화 (CPM 피팅) ---
            print(f"[{i+1}/{n_splits}] 2단계: 기술적 지표 파라미터 최적화 (Train 데이터 기준)...")
            train_df_with_signals, val_df_with_signals, test_df_with_signals, technical_signals_for_esn = ts_optimization(
                train_df_with_cpm, val_df_with_cpm, test_df_with_cpm
            )
            print(f"    -> 사용할 ESN 피처: {technical_signals_for_esn}")

            # --- 3단계: ESN 하이퍼파라미터 최적화 (GA) ---
            print(f"\n[{i+1}/{n_splits}] 3단계: ESN + 리스크 파라미터 최적화 (Train->Valid)...")
            best_params, _ = run_genetic_algorithm(
                train_df_ga=train_df_with_signals,
                test_df_ga=val_df_with_signals, # ⬅️ Validation Set 전달
                technical_signals_list=technical_signals_for_esn,
                pop_size=pop_size,
                num_generations=num_generations,
                random_seed=random_seed # ### 수정 2-1과 연결 ###
            )
            best_params_per_fold.append(best_params)
            print(f"[{i+1}/{n_splits}] ESN+리스크 최적 파라미터 (Valid 기준): {best_params}")

            # --- 4단계: 최종 성능 평가 ---
            print(f"\n[{i+1}/{n_splits}] 4단계: 최종 백테스팅 ( (Train+Valid)->Test )...")
            
            # 4a. Train+Valid 데이터 합치기
            final_train_df = pd.concat([train_df_with_signals, val_df_with_signals])
            final_test_df = test_df_with_signals
            
            # 4b. GA가 보지 못한 Test 데이터로 최종 백테스트
            stats, _ = perform_final_backtest(
                train_df=final_train_df,       # ⬅️ Train+Valid 신호 데이터
                test_df=final_test_df,         # ⬅️ Test 신호 데이터
                best_params=best_params,       # ⬅️ 3단계에서 찾은 최적 파라미터
                technical_signals_list=technical_signals_for_esn,
                random_state=random_seed,
                commission=commission,  # [추가] 전달
                slippage=slippage       # [추가] 전달
            )
            
            if stats is not None:
                print(f"\n--- 폴드 {i+1} 최종 성과 (Test Set) ---")
                print(f"Return [%]: {stats['Return [%]']:.2f}")
                print(f"Buy & Hold Return [%]: {stats['Buy & Hold Return [%]']:.2f}")
                print(f"Max. Drawdown [%]: {stats['Max. Drawdown [%]']:.2f}")
                print(f"Sharpe Ratio: {stats['Sharpe Ratio']:.2f}")
                
                total_returns.append(stats['Return [%]'])
                bh_returns.append(stats['Buy & Hold Return [%]'])
                total_mdd.append(stats['Max. Drawdown [%]'])
                total_sharpe.append(stats['Sharpe Ratio'])
        
        except Exception as e:
            print(f"폴드 {i+1} 처리 중 치명적 오류 발생: {e}")
            traceback.print_exc()

    print("\n" + "="*50)
    print("롤링 포워드 교차 검증 최종 결과 (Train/Validation/Test 분리):")
    if total_returns:
        print(f"총 {len(total_returns)}개 폴드 결과")
        print(f"각 폴드 Return [%] (Test Set): {[round(r, 2) for r in total_returns]}")
        print(f"평균 Return [%]: {np.mean(total_returns):.4f}")
        print(f"Buy&Hold 평균 Return [%]: {np.mean(bh_returns):.4f}")
        print(f"평균 MDD [%]: {np.mean(total_mdd):.4f}")
        print(f"평균 Sharpe Ratio: {np.mean(total_sharpe):.4f}")
    else:
        print("유효한 백테스팅 결과가 없습니다.")
    print("="*50)
    
    return best_params_per_fold, total_returns, None