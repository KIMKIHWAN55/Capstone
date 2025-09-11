import numpy as np
import pandas as pd
from bisect import insort_left

def extract_local_extrema(prices_series):
    critical_points = []
    if len(prices_series) < 2:
        return critical_points

    # 값을 float()로 명시적으로 변환하여 unhashable 오류 방지
    critical_points.append((prices_series.index[0], float(prices_series.iloc[0])))
    prices_values = prices_series.values
    indices = prices_series.index

    for i in range(1, len(prices_values) - 1):
        if (prices_values[i] > prices_values[i-1] and
            prices_values[i] > prices_values[i+1]):
            critical_points.append((indices[i], float(prices_values[i])))
        elif (prices_values[i] < prices_values[i-1] and
              prices_values[i] < prices_values[i+1]):
            critical_points.append((indices[i], float(prices_values[i])))

    critical_points.append((prices_series.index[-1], float(prices_series.iloc[-1])))
    
    critical_points = sorted(list(set(critical_points)), key=lambda x: x[0])
    return critical_points

def calculate_oscillation(y1, y2, epsilon=np.finfo(float).eps):
    denominator = (y1 + y2) / 2
    if abs(denominator) < epsilon:
        return float('inf')
    return abs(y2 - y1) / denominator

def calculate_duration(x1, x2):
    if isinstance(x1, (pd.Timestamp, pd.DatetimeIndex)):
        return abs((x2 - x1).days)
    return abs(x2 - x1)

def cpm_processing(critical_points, P, T):
    if len(critical_points) < 3:
        return critical_points

    selected_critical_points = [critical_points[0]]
    selected_set = {critical_points[0]}

    def add_point_to_selected(point):
        if point not in selected_set:
            insort_left(selected_critical_points, point, key=lambda x: x[0])
            selected_set.add(point)

    p_i_idx = 0
    p_j_idx = 1
    p_k_idx = 2

    while True:
        if p_k_idx >= len(critical_points):
            for idx in range(p_i_idx, len(critical_points)):
                add_point_to_selected(critical_points[idx])
            break

        prev_p_i_idx, prev_p_j_idx, prev_p_k_idx = p_i_idx, p_j_idx, p_k_idx

        i_point = critical_points[p_i_idx]
        j_point = critical_points[p_j_idx]
        k_point = critical_points[p_k_idx]

        x_i, y_i = i_point
        x_j, y_j = j_point
        x_k, y_k = k_point

        osc_ij = calculate_oscillation(y_i, y_j)
        dur_ij = calculate_duration(x_i, x_j)
        osc_jk = calculate_oscillation(y_j, y_k)
        dur_jk = calculate_duration(x_j, x_k)

        is_dur_ij_long = dur_ij >= T
        is_dur_jk_long = dur_jk >= T

        osc_rise = 0.0
        osc_decline = 0.0

        if y_i < y_j and y_j > y_k:
            osc_rise = osc_ij
            osc_decline = osc_jk
        elif y_i > y_j and y_j < y_k:
            osc_decline = osc_ij
            osc_rise = osc_jk

        is_rise_over_P = osc_rise >= P
        is_decline_over_P = osc_decline >= P

        current_case = 0
        if is_rise_over_P and is_decline_over_P:
            current_case = 1
        elif is_rise_over_P and not is_decline_over_P:
            current_case = 2
        elif not is_rise_over_P and is_decline_over_P:
            current_case = 3
        else:
            current_case = 4

        next_p_i_idx, next_p_j_idx, next_p_k_idx = p_i_idx, p_j_idx, p_k_idx

        if is_dur_ij_long:
            add_point_to_selected(i_point)
            add_point_to_selected(j_point)
            next_p_i_idx = p_i_idx + 2
            next_p_j_idx = p_i_idx + 3
            next_p_k_idx = p_i_idx + 4
        else:
            if current_case == 1:
                add_point_to_selected(i_point)
                add_point_to_selected(j_point)
                next_p_i_idx = p_i_idx + 2
                next_p_j_idx = p_i_idx + 3
                next_p_k_idx = p_i_idx + 4
    
            if current_case == 2:
                if p_k_idx + 1 < len(critical_points):
                    i_plus_3_point = critical_points[p_k_idx + 1]
                    y_i3 = i_plus_3_point[1]
        
                    if y_i < y_j and y_j > y_k:
                        if y_i3 >= y_j:
                            next_p_i_idx = p_i_idx
                            next_p_j_idx = p_k_idx + 1
                            next_p_k_idx = p_k_idx + 2
                        else:
                            next_p_i_idx = p_i_idx
                            next_p_j_idx = p_j_idx
                            next_p_k_idx = p_k_idx + 2
                    else:
                        if y_i3 >= y_j:
                            next_p_i_idx = p_i_idx
                            next_p_j_idx = p_j_idx
                            next_p_k_idx = p_k_idx + 2
                        else:
                            next_p_i_idx = p_i_idx
                            next_p_j_idx = p_k_idx + 1
                            next_p_k_idx = p_k_idx + 2
                else:
                    next_p_i_idx = p_i_idx + 1
                    next_p_j_idx = p_i_idx + 2
                    next_p_k_idx = p_i_idx + 3
    
            elif current_case == 3:
                next_p_i_idx = p_k_idx
                next_p_j_idx = p_k_idx + 1
                next_p_k_idx = p_k_idx + 2
    
            else:
                if is_dur_jk_long:
                    add_point_to_selected(i_point)
                    add_point_to_selected(k_point)
                    next_p_i_idx = p_k_idx
                    next_p_j_idx = p_k_idx + 1
                    next_p_k_idx = p_k_idx + 2
                else:
                    if y_i <= y_k:
                        next_p_i_idx = p_i_idx
                        next_p_j_idx = p_k_idx + 1
                        next_p_k_idx = p_k_idx + 2
                    else:
                        next_p_i_idx = p_k_idx
                        next_p_j_idx = p_k_idx + 1
                        next_p_k_idx = p_k_idx + 2

        p_i_idx = next_p_i_idx
        p_j_idx = next_p_j_idx
        p_k_idx = next_p_k_idx

        if (p_i_idx == prev_p_i_idx and p_j_idx == prev_p_j_idx and p_k_idx == prev_p_k_idx):
            for idx in range(p_i_idx, len(critical_points)):
                add_point_to_selected(critical_points[idx])
            break

    add_point_to_selected(critical_points[-1])
    return sorted(list(selected_set), key=lambda x: x[0])

def cpm_model(data, column='Close', P=0.05, T=5):
    original_df = data.copy()
    prices_series = data[column] if isinstance(data, pd.DataFrame) else data
    all_critical_points = extract_local_extrema(prices_series)
    processed_critical_points = cpm_processing(all_critical_points, P, T)
    cpm_point_indices = [point[0] for point in processed_critical_points]
    original_df['is_cpm_point'] = original_df.index.isin(cpm_point_indices)
    all_extrema_dict = {}
    prices_values = prices_series.values
    indices = prices_series.index
    for i in range(1, len(prices_values) - 1):
        if prices_values[i] > prices_values[i-1] and prices_values[i] > prices_values[i+1]:
            all_extrema_dict[indices[i]] = 1
        elif prices_values[i] < prices_values[i-1] and prices_values[i] < prices_values[i+1]:
            all_extrema_dict[indices[i]] = -1
    all_extrema_dict[prices_series.index[0]] = 0
    all_extrema_dict[prices_series.index[-1]] = 0
    original_df['cpm_point_type'] = 0
    for cpm_idx, _ in processed_critical_points:
        if cpm_idx in all_extrema_dict:
            original_df.loc[cpm_idx, 'cpm_point_type'] = all_extrema_dict[cpm_idx]
    return processed_critical_points, original_df