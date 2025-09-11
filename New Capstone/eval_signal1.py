# 파일명: eval_signal1.py (이제 백테스팅 및 샤프 지수 계산을 담당)

import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
import talib

# 범용 신호 처리 스트래те지
class SignalStrategy(Strategy):
    atr_period = 14
    atr_multiplier = 2.0  # 기본값, GA가 덮어쓸 수 있음
    tp_ratio = 1.5        # 기본값, GA가 덮어쓸 수 있음

    def init(self):
        # 데이터프레임의 'Signal' 컬럼을 신호로 사용
        self.signal = self.I(lambda x: x, self.data.Signal, name='signal')
        self.atr = self.I(talib.ATR, 
                          self.data.High, self.data.Low, self.data.Close, 
                          timeperiod=self.atr_period, name="ATR")

    def next(self):
        # -1을 매수 신호, 1을 매도(청산) 신호로 사용
        if self.signal[-1] == -1 and not self.position:
            sl_price = self.data.Close[-1] - (self.atr_multiplier * self.atr[-1])
            stop_loss_distance = self.data.Close[-1] - sl_price
            tp_price = self.data.Close[-1] + (stop_loss_distance * self.tp_ratio)
            self.buy(sl=sl_price, tp=tp_price)
        elif self.signal[-1] == 1 and self.position.is_long:
            self.position.close()

# 범용 피트니스 계산 함수 (이제 이 파일의 핵심 기능)
def calculate_fitness_with_backtest(df_with_signals, atr_multiplier=2.0, tp_ratio=1.5):
    # 'Signal' 컬럼이 없으면 페널티
    if 'Signal' not in df_with_signals.columns or df_with_signals['Signal'].eq(-1).sum() == 0:
        return -1000.0,

    try:
        bt = Backtest(df_with_signals, SignalStrategy, cash=10000, commission=.002, exclusive_orders=True)
        stats = bt.run(atr_multiplier=atr_multiplier, tp_ratio=tp_ratio)
    except Exception:
        return -1000.0, # 오류 발생 시 페널티

    sharpe_ratio = stats['Sharpe Ratio']
    if pd.isna(sharpe_ratio) or np.isinf(sharpe_ratio) or sharpe_ratio < -5:
        return -1000.0,

    max_drawdown = abs(stats['Max. Drawdown [%]'])
    if max_drawdown > 25:
        penalty_factor = max_drawdown / 100.0
        fitness = sharpe_ratio * (1 - penalty_factor)
    else:
        fitness = sharpe_ratio
    
    return fitness,