# ESN_Signals.py (CV_ESN3.py 호환 수정본)
#
# [수정] ESN 학습 타겟을 'MACD_Trend'에서 'cpm_point_type'으로 변경
#        (CV_ESN3.py의 1단계 CPM 정답 생성을 ESN 타겟으로 사용)

import pandas as pd
from sklearn.preprocessing import StandardScaler
from pyESN import ESN
import numpy as np

def t_softmax(logits, temperature=1.0):
    """ T-Softmax 함수: ESN의 원시 출력(logits)을 확률로 변환합니다. """
    # temperature가 0에 가까우면 overflow 방지를 위해 매우 작은 값으로 대체
    if temperature < 1e-6:
        temperature = 1e-6
        
    # logits / T
    logits_scaled = logits / temperature
    
    # 안정적인 Softmax 계산 (Max-Subtraction 트릭)
    logits_scaled = logits_scaled - np.max(logits_scaled, axis=1, keepdims=True)
    
    exps = np.exp(logits_scaled)
    sum_exps = np.sum(exps, axis=1, keepdims=True)
    
    # 0으로 나누기 방지
    sum_exps[sum_exps == 0] = 1e-9
    
    return exps / sum_exps

def esn_signals(train_df: pd.DataFrame, test_df: pd.DataFrame, Technical_Signals: list,
                n_reservoir: int = 200, spectral_radius: float = 0.95, sparsity: float = 0.1,
                random_state: int = 42,
                # --- GA 튜닝 인자 추가 ---
                buy_threshold: float = 0.5,  # (하위 호환용, th_buy/th_sell 사용 시 무시됨)
                sell_threshold: float = 0.5, # (하위 호환용, th_buy/th_sell 사용 시 무시됨)
                signal_threshold: float = 0.5, # (ESN_GA.py 호환용)
                th_buy: float = 0.5,           # (ESN_GA.py 호환용 - 매수 임계값)
                th_sell: float = 0.5,          # (ESN_GA.py 호환용 - 매도 임계값)
                temp_T: float = 1.0,           # (ESN_GA.py 호환용 - T-Softmax 온도)
                min_hold: int = 0,             # (ESN_GA.py 호환용 - 현재 미사용)
                cooldown: int = 0,             # (ESN_GA.py 호환용 - 현재 미사용)
                **kwargs):                     # 향후 추가될 인자를 위한 여유분
    """
    ESN 모델을 학습하고 매수/매도 신호를 생성합니다. (3-Class Softmax 버전)

    Args:
        train_df (pd.DataFrame): ESN 학습에 사용할 학습 데이터 (Technical_Signals 포함).
        test_df (pd.DataFrame): ESN이 신호를 생성할 테스트 데이터 (Technical_Signals 포함).
        Technical_Signals (list): ESN 입력으로 사용할 기술적 신호 컬럼 이름 리스트.
        n_reservoir (int): ESN의 Reservoir 크기.
        spectral_radius (float): Reservoir의 Spectral Radius.
        sparsity (float): Reservoir의 희소성.
        random_state (int): 난수 시드.
        th_buy (float): T-Softmax 확률 기반 매수 임계값 (ESN_GA.py 연동)
        th_sell (float): T-Softmax 확률 기반 매도 임계값 (ESN_GA.py 연동)
        temp_T (float): T-Softmax의 온도 (낮을수록 확률이 뾰족해짐)

    Returns:
        pd.DataFrame: ESN 모델이 test_df에 대해 생성한 매수/매도 신호 DataFrame.
                      'Close', 'Predicted_Signals' 컬럼을 포함합니다.
    """

    # 1. 학습 데이터 준비
    train_df_copy = train_df.copy()
    train_df_copy['Close_p'] = train_df_copy['Close'].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
    test_df_copy = test_df.copy()
    test_df_copy['Close_p'] = test_df_copy['Close'].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
    features = Technical_Signals + ['Close_p']
    
    # --- [수정] CV_ESN3.py의 의도에 맞게 타겟 변수 변경 ---
    # 기존: MACD 부호를 타겟으로 사용
    # train_df_copy['MACD_Trend'] = np.sign(train_df_copy['MACD_Value']).astype(int)
    # train_df_copy['Target_cpm_point_type'] = train_df_copy['MACD_Trend'].shift(-1)

    # 변경: CV_ESN3.py가 1단계에서 생성한 cpm_point_type을 타겟으로 사용
    #       (ESN이 "다음 날"의 CPM 포인트를 예측하도록 shift(-1) 적용)
    train_df_copy['Target_cpm_point_type'] = train_df_copy['cpm_point_type'].shift(-1)
    # --------------------------------------------------

    # --- [수정] NaHandling 로직 변경 ---
    # 1. 피처(features)에 NaN이 있는 행을 먼저 제거 (SMA 200 등으로 인한 앞부분 제거)
    #    (CV_ESN3.py의 generate_signals에서 이미 NaN을 0, 50 등으로 채우므로 주석 처리)
    #df_esn_train = train_df_copy.dropna(subset=features)
    df_esn_train = train_df_copy
    
    # 2. 타겟(Target_cpm_point_type)에 NaN이 있는 행을 제거
    #    (shift(-1)로 인한 마지막 행 제거 + cpm_point_type 자체의 NaN 제거)
    df_esn_train = df_esn_train.dropna(subset=['Target_cpm_point_type'])
    # --- [수정 완료] ---

    # 학습 데이터가 충분하지 않을 경우 즉시 빈 DataFrame 반환
    if df_esn_train.empty:
        # print("ESN_Signals: df_esn_train이 비어있습니다. (NaN 처리 후)") # 디버깅 로그
        return pd.DataFrame(columns=['Close', 'Predicted_Signals'])

    X_train_raw = df_esn_train[features].values
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    
    # (수정 2: y_train을 One-Hot 인코딩으로 변경)
    # Target: {-1: 매수(저점), 0: 보류, 1: 매도(고점)}
    # One-Hot: {매수: [1,0,0], 보류: [0,1,0], 매도: [0,0,1]}
    y_train_raw = df_esn_train['Target_cpm_point_type'].values.astype(int)
    y_train = np.zeros((len(y_train_raw), 3))
    y_train[y_train_raw == -1, 0] = 1  # 매수 (Index 0)
    y_train[y_train_raw == 0,  1] = 1  # 보류 (Index 1)
    y_train[y_train_raw == 1,  2] = 1  # 매도 (Index 2)

    # 2. ESN 모델 초기화 및 학습
    n_inputs = X_train.shape[1]
    n_outputs = 3  # (수정 3: 출력을 1개에서 3개로 변경)

    esn_model = ESN(n_inputs=n_inputs, n_outputs=n_outputs, n_reservoir=n_reservoir,
                    spectral_radius=spectral_radius, sparsity=sparsity,
                    input_scaling=1.0, 
                    teacher_scaling=1.0, 
                    teacher_shift=0.0,
                    random_state=random_state, silent=True)

    esn_model.fit(X_train, y_train)

    # 3. 테스트 데이터 준비 및 예측
    # 예측에 사용할 피처 데이터 준비
    # (CV_ESN3.py의 generate_signals에서 이미 NaN을 0, 50 등으로 채우므로 주석 처리)
    #df_esn_test = test_df_copy.dropna(subset=features) 
    df_esn_test = test_df_copy

    # 테스트 데이터가 충분하지 않을 경우 즉시 빈 DataFrame 반환
    if df_esn_test.empty:
        return pd.DataFrame(columns=['Close', 'Predicted_Signals'])

    X_test_raw = df_esn_test[features].values
    X_test = scaler.transform(X_test_raw)

    test_indices = df_esn_test.index
    test_close_prices = df_esn_test['Close']

    # esn_predictions는 이제 (n_samples, 3) 형태의 로짓(logits)이 됩니다.
    esn_predictions_logits = esn_model.predict(X_test)

    # (수정 4: T-Softmax 적용 및 임계값 기반 신호 생성)
    # T-Softmax를 적용하여 확률로 변환
    esn_probs = t_softmax(esn_predictions_logits, temperature=temp_T)
    
    buy_probs = esn_probs[:, 0]  # 매수 확률 (Index 0)
    hold_probs = esn_probs[:, 1] # 보류 확률 (Index 1)
    sell_probs = esn_probs[:, 2] # 매도 확률 (Index 2)

    # 4. 예측값을 매수/매도 신호로 변환
    esn_signals_df = pd.DataFrame(index=test_indices)
    esn_signals_df['Buy_Prob'] = buy_probs
    esn_signals_df['Sell_Prob'] = sell_probs
    esn_signals_df['Close'] = test_close_prices

    esn_signals_df['Type_Num'] = 0 # 기본값 HOLD (0)
    
    # 확률이 임계값을 넘고, 해당 확률이 가장 높을 때만 신호 발생
    is_sell_signal = (sell_probs > th_sell) & (sell_probs > buy_probs) & (sell_probs > hold_probs)
    is_buy_signal = (buy_probs > th_buy) & (buy_probs > sell_probs) & (buy_probs > hold_probs)

    esn_signals_df.loc[is_sell_signal, 'Type_Num'] = 1  # SELL (고점 예측)
    esn_signals_df.loc[is_buy_signal, 'Type_Num'] = -1 # BUY (저점 예측)

    # 백테스팅을 위해 필요한 컬럼만 추출
    backtest_signals = esn_signals_df[['Close', 'Type_Num']].rename(columns={'Type_Num': 'Predicted_Signals'})

    return backtest_signals