import numpy as np
import pandas as pd

def extract_local_extrema(prices_series):
    """
    가격 시리즈에서 모든 국소 극점(최대/최소)을 식별합니다.
    시작점과 끝점을 포함하며, 각 점의 유형(1: max, -1: min, 0: N/A)을 반환합니다.
    """
    critical_points = []
    if len(prices_series) < 2:
        return critical_points

    prices_values = prices_series.values
    indices = prices_series.index

    # 시작점 (유형: 다음 포인트와의 관계에 따라 결정)
    start_type = 0
    if len(prices_values) > 1:
        if prices_values[1] > prices_values[0]:
            start_type = -1  # 시작이 저점
        elif prices_values[1] < prices_values[0]:
            start_type = 1   # 시작이 고점
    critical_points.append((indices[0], prices_values[0], start_type))

    # 중간 극점
    for i in range(1, len(prices_values) - 1):
        if prices_values[i] > prices_values[i-1] and prices_values[i] > prices_values[i+1]:
            critical_points.append((indices[i], prices_values[i], 1))  # 최대점
        elif prices_values[i] < prices_values[i-1] and prices_values[i] < prices_values[i+1]:
            critical_points.append((indices[i], prices_values[i], -1)) # 최소점

    # 끝점 (유형: 이전 포인트와의 관계에 따라 결정)
    end_type = 0
    if len(prices_values) > 1:
        if prices_values[-1] > prices_values[-2]:
            end_type = 1   # 끝이 고점
        elif prices_values[-1] < prices_values[-2]:
            end_type = -1  # 끝이 저점
    critical_points.append((indices[-1], prices_values[-1], end_type))

    # 중복 제거 및 정렬 (거의 발생하지 않지만 안전장치)
    unique_points = sorted(list(set(critical_points)), key=lambda x: x[0])
    return unique_points

def calculate_oscillation(y1, y2, epsilon=np.finfo(float).eps):
    """ 두 점 사이의 진폭(변동률)을 계산합니다. """
    denominator = (y1 + y2) / 2
    if abs(denominator) < epsilon:
        return float('inf')  # 0으로 나누기 방지
    return abs(y2 - y1) / denominator

def calculate_duration(x1, x2):
    """ 두 점 사이의 시간(기간)을 계산합니다. """
    if isinstance(x1, (pd.Timestamp, pd.DatetimeIndex)):
        return abs((x2 - x1).days)
    return abs(x2 - x1)

def cpm_processing(critical_points, P, T):
    """
    논문의 4가지 Case를 기반으로 임계점을 처리(필터링)합니다.
    """
    if len(critical_points) < 3:
        return critical_points

    # 선택된 포인트를 저장하는 리스트 (시작점은 항상 포함)
    selected_critical_points = [critical_points[0]]
    # 중복 추가를 방지하기 위한 set
    selected_set = {critical_points[0]}

    def add_point_to_selected(point):
        """ 
        중복만 체크하고 리스트 끝에 추가합니다. 
        이미 시간순으로 처리되므로 정렬이 유지됩니다.
        """
        if point not in selected_set:
            selected_critical_points.append(point) # 'append' 사용
            selected_set.add(point)

    p_i_idx = 0  # 테스트 유닛의 첫 번째 점 (i)
    p_j_idx = 1  # 테스트 유닛의 두 번째 점 (j)
    p_k_idx = 2  # 테스트 유닛의 세 번째 점 (k)

    while True:
        # k가 리스트 범위를 벗어나면, 남은 점들을 모두 추가하고 종료
        if p_k_idx >= len(critical_points):
            for idx in range(p_i_idx, len(critical_points)):
                add_point_to_selected(critical_points[idx])
            break

        prev_p_i_idx, prev_p_j_idx, prev_p_k_idx = p_i_idx, p_j_idx, p_k_idx

        i_point = critical_points[p_i_idx]
        j_point = critical_points[p_j_idx]
        k_point = critical_points[p_k_idx]

        x_i, y_i, type_i = i_point
        x_j, y_j, type_j = j_point
        x_k, y_k, type_k = k_point

        osc_ij = calculate_oscillation(y_i, y_j)
        dur_ij = calculate_duration(x_i, x_j)
        osc_jk = calculate_oscillation(y_j, y_k)
        dur_jk = calculate_duration(x_j, x_k)

        # [T와 P 통합]
        is_ij_significant = (osc_ij >= P) or (dur_ij >= T)
        is_jk_significant = (osc_jk >= P) or (dur_jk >= T)

        # Case 분류
        current_case = 0
        if is_ij_significant and is_jk_significant:
            current_case = 1  # (Big, Big)
        elif is_ij_significant and not is_jk_significant:
            current_case = 2  # (Big, Small) - Look Ahead 필요
        elif not is_ij_significant and is_jk_significant:
            current_case = 3  # (Small, Big) - j 폐기
        else:
            current_case = 4  # (Small, Small) - j 폐기

        next_p_i_idx, next_p_j_idx, next_p_k_idx = p_i_idx, p_j_idx, p_k_idx

        if current_case == 1:
            # (Big, Big) -> i, j 모두 저장. 다음 유닛 (j, k, l)
            add_point_to_selected(i_point)
            add_point_to_selected(j_point)
            next_p_i_idx = p_j_idx
            next_p_j_idx = p_k_idx
            next_p_k_idx = p_k_idx + 1

        elif current_case == 2:
            # (Big, Small) -> Look Ahead (Fig. 10)
            p_l_idx = p_k_idx + 1
            if p_l_idx >= len(critical_points):
                # 'look ahead' 할 점이 없음. Case 1처럼 j를 보존
                add_point_to_selected(i_point)
                add_point_to_selected(j_point)
                next_p_i_idx = p_j_idx
                next_p_j_idx = p_k_idx
                next_p_k_idx = p_l_idx
            else:
                l_point = critical_points[p_l_idx]
                y_l, type_l = l_point[1], l_point[2]

                is_j_dumped = False
                # j가 고점(1)이고, 다음 고점(l)이 j보다 높거나 같으면 j 폐기
                if type_j == 1 and type_l == 1 and y_l >= y_j:
                    is_j_dumped = True
                # j가 저점(-1)이고, 다음 저점(l)이 j보다 낮거나 같으면 j 폐기
                elif type_j == -1 and type_l == -1 and y_l <= y_j:
                    is_j_dumped = True

                if is_j_dumped:
                    # j 폐기. i는 저장. 다음 유닛 (i, k, l)
                    add_point_to_selected(i_point)
                    next_p_i_idx = p_i_idx
                    next_p_j_idx = p_k_idx
                    next_p_k_idx = p_l_idx
                else:
                    # j 보존. i, j 저장. 다음 유닛 (j, k, l)
                    add_point_to_selected(i_point)
                    add_point_to_selected(j_point)
                    next_p_i_idx = p_j_idx
                    next_p_j_idx = p_k_idx
                    next_p_k_idx = p_l_idx

        elif current_case == 3:
            # (Small, Big) -> j 폐기 (Fig. 11)
            # i 저장. 다음 유닛 (i, k, l)
            add_point_to_selected(i_point)
            next_p_i_idx = p_i_idx
            next_p_j_idx = p_k_idx
            next_p_k_idx = p_k_idx + 1

        elif current_case == 4:
            # (Small, Small) -> j 폐기 (Fig. 12)
            # i 저장. 다음 유닛 (i, k, l)
            add_point_to_selected(i_point)
            next_p_i_idx = p_i_idx
            next_p_j_idx = p_k_idx
            next_p_k_idx = p_k_idx + 1

        p_i_idx = next_p_i_idx
        p_j_idx = next_p_j_idx
        p_k_idx = next_p_k_idx

        # 무한 루프 방지 (인덱스가 더 이상 진행되지 않을 때)
        if (p_i_idx == prev_p_i_idx and 
            p_j_idx == prev_p_j_idx and 
            p_k_idx == prev_p_k_idx):
            # 남은 점들을 모두 추가하고 강제 종료
            for idx in range(p_i_idx, len(critical_points)):
                add_point_to_selected(critical_points[idx])
            break

    # 마지막 점은 항상 추가 (루프에서 빠졌을 경우 대비)
    add_point_to_selected(critical_points[-1])
    
    return selected_critical_points

def cpm_model(data, column='Close', P=0.05, T=5):
    """
    데이터프레임 또는 시리즈를 입력받아 CPM을 적용하고,
    원본 데이터프레임에 'is_cpm_point'와 'cpm_point_type' 열을 추가하여 반환합니다.
    """
    if isinstance(data, pd.DataFrame):
        original_df = data.copy()
        prices_series = data[column]
    else:
        prices_series = data
        original_df = pd.DataFrame(prices_series, columns=[column])

    all_critical_points = extract_local_extrema(prices_series)
    
    # CPM 처리
    processed_critical_points = cpm_processing(all_critical_points, P, T)
    
    # 결과 매핑
    cpm_point_dict = {point[0]: point[2] for point in processed_critical_points}
    
    original_df['is_cpm_point'] = original_df.index.isin(cpm_point_dict.keys())
    
    # cpm_point_type: 1(max), -1(min), 0(cpm point 아님 또는 시작/끝점)
    original_df['cpm_point_type'] = original_df.index.map(cpm_point_dict).fillna(0).astype(int)

    # 처리된 (index, value, type) 튜플 리스트와 업데이트된 DF 반환
    return processed_critical_points, original_df

