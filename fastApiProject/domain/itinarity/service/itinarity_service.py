# domain/itinarity/service/itinarity_service.py
import random
import re
from typing import List, Dict, Any

import numpy as np
import pulp


##################################
# 유틸리티/보조 함수 정의
##################################

def calculate_distance(p1: tuple, p2: tuple) -> float:
    """
    두 지점(p1, p2)의 (x, y) 좌표를 받아 유클리드 거리를 계산.
    """
    (x1, y1) = p1
    (x2, y2) = p2
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def estimate_travel_cost(dist: float, transport_mode: str = 'car') -> float:
    """
    주어진 거리와 교통수단에 따라 이동 비용(또는 점수 감점)을 추정.
    """
    if transport_mode == 'car':
        return dist * 1000  # 예: 1km 당 1000원
    elif transport_mode == 'public':
        return dist * 500  # 대중교통은 더 저렴하게
    return dist * 800  # 기타 교통수단


def user_satisfaction_score(place_info: Dict[str, Any], user_profile: Dict[str, Any]) -> float:
    """
    장소(place_info)와 사용자 프로필(user_profile)을 기반으로
    해당 장소 방문 시 얻을 만족도를 산출.
    """
    base_score = place_info.get('base_score', 10)
    theme_score = 0
    if 'themes' in user_profile:
        if place_info.get('category') in user_profile['themes']:
            theme_score += 5
    must_visit_list = user_profile.get('must_visit_list', [])
    if place_info['id'] in must_visit_list:
        theme_score += 10
    return base_score + theme_score


##################################
# MIP 기반 최적화 함수
##################################

def solve_itinerary_mip(places: List[Dict[str, Any]], user_profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    혼합정수계획(MIP)을 이용하여 주어진 후보 관광지(places)와
    사용자 프로필(user_profile)을 기반으로 최적의 방문 경로를 산출.

    반환값:
      - status: 최적화 문제 상태 (예: "Optimal")
      - objective: 목적 함수 최종 값 (만족도 점수 - 이동 비용)
      - route_indices: (순서, 관광지 인덱스) 쌍 리스트
      - route_place_ids: 최종 선택된 관광지의 ID 순서 리스트
    """
    n = len(places)

    # 거리 및 비용 행렬 계산
    dist_matrix = np.zeros((n, n))
    cost_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                dist_matrix[i, j] = 0
                cost_matrix[i, j] = 999999  # 자기 자신 이동은 제외
            else:
                dist = calculate_distance((places[i]['x'], places[i]['y']),
                                          (places[j]['x'], places[j]['y']))
                c = estimate_travel_cost(dist, user_profile.get('preferred_transport', 'car'))
                dist_matrix[i, j] = dist
                cost_matrix[i, j] = c

    # 각 장소의 만족도 계산
    sat_scores = [user_satisfaction_score(places[i], user_profile) for i in range(n)]

    # PuLP 문제 정의: 최대화 문제
    prob = pulp.LpProblem("Itinerary_Optimization", pulp.LpMaximize)

    # 결정 변수 정의
    x = pulp.LpVariable.dicts('x', (range(n), range(n)), cat=pulp.LpBinary)
    y = pulp.LpVariable.dicts('y', (range(n), range(n)), cat=pulp.LpBinary)
    u = pulp.LpVariable.dicts('u', range(n), lowBound=0, upBound=n, cat=pulp.LpInteger)

    # 목적 함수: 총 만족도 점수 - 총 이동 비용
    prob += (
            pulp.lpSum([sat_scores[i] * y[k][i] for i in range(n) for k in range(n)])
            - pulp.lpSum([cost_matrix[i][j] * x[i][j] for i in range(n) for j in range(n)])
    )

    # 제약 조건 1: 각 노드는 최대 한 번 방문
    for i in range(n):
        prob += pulp.lpSum([y[k][i] for k in range(n)]) <= 1

    # 제약 조건 2: 들어오는 경로와 나가는 경로의 일관성 유지
    for i in range(n):
        prob += pulp.lpSum([x[j][i] for j in range(n) if j != i]) == pulp.lpSum([y[k][i] for k in range(n)])
        prob += pulp.lpSum([x[i][j] for j in range(n) if j != i]) == pulp.lpSum([y[k][i] for k in range(n)])

    # 제약 조건 3: Subtour 방지 (MTZ 제약식, 간략화된 형태)
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                prob += u[i] - u[j] + n * x[i][j] <= n - 1

    # 예산 제약: 총 이동 비용 및 입장료 합 <= 예산
    if 'budget_amount' in user_profile:
        total_cost_expr = pulp.lpSum([cost_matrix[i][j] * x[i][j] for i in range(n) for j in range(n)])
        entry_fee_expr = pulp.lpSum(
            [places[i].get('entry_fee', 0) * pulp.lpSum([y[k][i] for k in range(n)]) for i in range(n)]
        )
        prob += (total_cost_expr + entry_fee_expr) <= user_profile['budget_amount']

    # 필수 방문지 제약: must_visit_list에 해당하는 장소는 반드시 방문
    must_visit_list = user_profile.get('must_visit_list', [])
    for mv_id in must_visit_list:
        for i in range(n):
            if places[i]['id'] == mv_id:
                prob += pulp.lpSum([y[k][i] for k in range(n)]) == 1

    # 최적화 문제 풀기 (CBC 솔버 사용)
    prob.solve(pulp.PULP_CBC_CMD(msg=1))

    status = pulp.LpStatus[prob.status]
    objective_value = pulp.value(prob.objective)

    # 해석: 각 순서별 방문 노드 추출
    best_route = []
    for k in range(n):
        for i in range(n):
            if pulp.value(y[k][i]) and pulp.value(y[k][i]) > 0.5:
                best_route.append((k, i))
    best_route.sort(key=lambda x: x[0])
    route_place_ids = [places[i]['id'] for (k, i) in best_route]

    return {
        'status': status,
        'objective': objective_value,
        'route_indices': best_route,
        'route_place_ids': route_place_ids
    }


##################################
# Tabu Search 기반 휴리스틱 함수
##################################

def tabu_search_itinerary(places: List[Dict[str, Any]], user_profile: Dict[str, Any],
                          max_iter: int = 1000) -> Dict[str, Any]:
    """
    간단한 Tabu Search(금기 탐색) 알고리즘을 사용하여 최적의 여행 경로를 산출.

    반환값:
      - best_route_places: 최적 경로 상의 관광지 이름 리스트
      - objective: 최적 경로의 목표 함수 값 (만족도 점수 - 이동 비용)
    """
    # 초기 해 구성: 필수 방문지를 우선 배치하고 나머지 후보를 무작위 섞음
    must_visit = user_profile.get('must_visit_list', [])
    route = [p for p in places if p['id'] in must_visit]
    others = [p for p in places if p['id'] not in must_visit]
    random.shuffle(others)
    insert_limit = min(5, len(others))
    route.extend(others[:insert_limit])

    tabu_list = []
    tabu_tenure = 10

    def calc_objective(route_):
        total_sat = 0
        total_cost = 0
        for i, place in enumerate(route_):
            total_sat += user_satisfaction_score(place, user_profile)
            if i < len(route_) - 1:
                dist = calculate_distance((place['x'], place['y']),
                                          (route_[i + 1]['x'], route_[i + 1]['y']))
                c = estimate_travel_cost(dist, user_profile.get('preferred_transport', 'car'))
                total_cost += c
        return total_sat - total_cost

    best_route = route[:]
    best_score = calc_objective(best_route)

    for iteration in range(max_iter):
        neighbor = best_route[:]
        if len(neighbor) > 1:
            i1, i2 = random.sample(range(len(neighbor)), 2)
            neighbor[i1], neighbor[i2] = neighbor[i2], neighbor[i1]
        neighbor_score = calc_objective(neighbor)
        if neighbor in tabu_list:
            continue
        if neighbor_score > best_score:
            best_route = neighbor
            best_score = neighbor_score
        tabu_list.append(neighbor)
        if len(tabu_list) > tabu_tenure:
            tabu_list.pop(0)

    return {
        'best_route_places': [p['name'] for p in best_route],
        'objective': best_score
    }