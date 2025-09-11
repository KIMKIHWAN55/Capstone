# 파일명: ESN_Signals.py (pyESN 라이브러리 사용 버전)

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pyESN import ESN 

def esn_signals(train_df, test_df, Technical_Signals, n_reservoir, spectral_radius, sparsity, signal_threshold, random_state=42):
    train_df_copy = train_df.copy()
    test_df_copy = test_df.copy()

    features = Technical_Signals
    
    # 1. Target(정답) 데이터 준비
    train_df_copy['Target'] = train_df_copy['cpm_point_type'].shift(-1).fillna(0)
    
    df_esn_train = train_df_copy.dropna(subset=features + ['Target'])
    df_esn_test = test_df_copy.dropna(subset=features)
    
    if df_esn_train.empty or df_esn_test.empty:
        return pd.DataFrame()
        
    X_train = df_esn_train[features].values
    y_train_raw = df_esn_train['Target'].values
    
    X_test = df_esn_test[features].values
    
    # 2. 데이터 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. Target을 One-Hot 인코딩으로 변환 (pyESN은 분류 문제를 회귀 문제로 풀어냄)
    # 클래스: -1, 0, 1 -> 3개의 출력 뉴런
    classes = np.array([-1, 0, 1])
    y_train_onehot = np.zeros((len(y_train_raw), len(classes)))
    for i, c in enumerate(classes):
        y_train_onehot[y_train_raw == c, i] = 1
        
    # 4. pyESN 모델 초기화 및 학습
    esn = ESN(
        n_inputs=len(features),
        n_outputs=len(classes),  # 클래스 개수만큼 출력 뉴런 설정
        n_reservoir=int(n_reservoir),
        spectral_radius=spectral_radius,
        sparsity=sparsity,
        random_state=random_state
    )
    
    esn.fit(X_train_scaled, y_train_onehot)
    
    predicted_output = esn.predict(X_test_scaled)

# --- 수정된 부분 시작 ---

# Softmax 함수를 적용하여 출력값을 확률처럼 변환 (0~1 사이, 총합 1)
# 이는 각 예측에 대한 모델의 '확신도'를 나타냅니다.
    def softmax(x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

    predicted_proba = softmax(predicted_output)

# 가장 확률이 높은 클래스를 예측 신호로 선택
    predicted_classes_indices = np.argmax(predicted_proba, axis=1)
    predicted_signals = classes[predicted_classes_indices]

# 가장 높은 확률값(모델의 확신도)을 가져옴
    max_proba = np.max(predicted_proba, axis=1)

# 모델의 확신도가 signal_threshold보다 낮으면 신호를 무시 (0으로 설정)
    predicted_signals[max_proba < signal_threshold] = 0


# --- 수정된 부분 끝 ---

    signals_df = pd.DataFrame({
        'Predicted_Signals': predicted_signals
    }, index=df_esn_test.index)

    return signals_df