# eval_signal3.py (전체 교체용 코드 - 정규화 적용)

import pandas as pd
import numpy as np

def calculate_total_fitness_optimized(
    data: pd.DataFrame,
    expected_trading_points: pd.DataFrame,
    suggested_signals: pd.DataFrame
) -> float:
    """
    이 함수는 '기대 거래 지점'(T_i)과 '제안된 신호'(S_j) 사이의
    가격 차이를 최소화하는 것을 목표로 합니다.
    (수정: 최종 오차를 n_expected로 정규화)

    Args:
        data: 'Close' 열과 정렬된 인덱스(날짜/시간)를 포함하는 전체 가격 데이터.
        expected_trading_points: 'Index' (T_i의 인덱스)와 'Type' ('BUY'/'SELL')을
                                 가진 DataFrame. (이상적인 거래 시점)
        suggested_signals: 'Index' (S_j의 인덱스), 'Type' ('BUY'/'SELL'), 'Close'
                           (S_j 시점의 가격)를 가진 DataFrame. (GA가 생성한 거래 시점)

    Returns:
        정규화된 총 피트니스 값 (낮을수록 좋음).
    """
    total_fitness = 0.0

    if expected_trading_points.empty:
        return 0.0

    # 1. 데이터 준비
    # (수정 5) 인덱스 정렬 강제
    data = data.sort_index()
    expected_trading_points_sorted = expected_trading_points.sort_values(by='Index')
    
    expected_indices = expected_trading_points_sorted['Index'].values
    expected_types = expected_trading_points_sorted['Type'].values
    
    data_indices_arr = data.index.values
    data_close_arr = data['Close'].values
    data_len = len(data_indices_arr)

    if not suggested_signals.empty:
        suggested_signals_sorted = suggested_signals.sort_values(by='Index')
        suggested_signal_indices_arr = suggested_signals_sorted['Index'].values
    else:
        suggested_signals_sorted = pd.DataFrame(columns=['Index', 'Type', 'Close'])
        suggested_signal_indices_arr = np.array([])

    n_expected = len(expected_indices)
    if n_expected == 0:
        return 0.0 # 정답이 없으면 오차 0

    # 2. 모든 기대 거래 지점(T_i)을 순회
    for i in range(n_expected):
        
        # 2-1. T_i, T_{i-1}, T_{i+1}의 '실제 인덱스'
        ti_index = expected_indices[i]
        ti_type = expected_types[i]
        ti_prev_index = expected_indices[max(0, i - 1)]
        ti_next_index = expected_indices[min(n_expected - 1, i + 1)]

        # 2-1B. 인덱스의 '정수 위치(iloc)' 찾기
        ti_iloc = np.searchsorted(data_indices_arr, ti_index)
        
        # 2-1C. KeyError 대신 명시적 확인
        if ti_iloc >= data_len or data_indices_arr[ti_iloc] != ti_index:
            continue
            
        ti_close = data_close_arr[ti_iloc]
        
        # 페널티 범위를 위한 정수 위치
        ti_prev_iloc = np.searchsorted(data_indices_arr, ti_prev_index, side='left')
        ti_next_iloc = np.searchsorted(data_indices_arr, ti_next_index, side='left')
        ti_next_iloc_inclusive = min(ti_next_iloc + 1, data_len)


        fitness_val = float('inf') 

        # 2-2. S_j 검색
        idx_start = np.searchsorted(suggested_signal_indices_arr, ti_prev_index, side='left')
        idx_end = np.searchsorted(suggested_signal_indices_arr, ti_next_index, side='right')
        
        relevant_sjs = suggested_signals_sorted.iloc[idx_start:idx_end]

        if not relevant_sjs.empty:
            closest_sj_idx_in_relevant = (relevant_sjs['Index'] - ti_index).abs().idxmin()
            Sj = relevant_sjs.loc[closest_sj_idx_in_relevant]
            Sj_close = Sj['Close']
            Sj_type = Sj['Type']

            # 2-4. 피트니스 계산 (Eq. 1, 2, 4, 5)
            if ti_type == 'BUY':
                if Sj_type == 'BUY':
                    fitness_val = Sj_close - ti_close
                elif Sj_type == 'SELL':
                    if abs(Sj_close - ti_close) / ti_close < 0.05:
                        penalty_slice_arr = data_close_arr[ti_iloc : ti_next_iloc_inclusive]
                        if penalty_slice_arr.size > 0:
                            max_price_in_range = np.max(penalty_slice_arr)
                            fitness_val = 2 * (max_price_in_range - ti_close)
            
            elif ti_type == 'SELL':
                if Sj_type == 'SELL':
                    fitness_val = ti_close - Sj_close
                elif Sj_type == 'BUY':
                    if abs(Sj_close - ti_close) / ti_close < 0.05:
                        penalty_slice_arr = data_close_arr[ti_iloc : ti_next_iloc_inclusive]
                        if penalty_slice_arr.size > 0:
                            min_price_in_range = np.min(penalty_slice_arr)
                            fitness_val = 2 * (ti_close - min_price_in_range)

        # 2-5. '놓친 신호' 처리
        if fitness_val == float('inf'):
            penalty_slice_arr = data_close_arr[ti_prev_iloc : ti_next_iloc_inclusive]
            
            if penalty_slice_arr.size > 0:
                if ti_type == 'BUY':
                    max_price_in_range = np.max(penalty_slice_arr)
                    fitness_val = max_price_in_range - ti_close
                elif ti_type == 'SELL':
                    min_price_in_range = np.min(penalty_slice_arr)
                    fitness_val = ti_close - min_price_in_range
            else:
                pass 

        total_fitness += fitness_val

    # --- [수정 4] 총 오차를 정답 개수로 나누어 정규화 ---
    return total_fitness / max(1, n_expected)