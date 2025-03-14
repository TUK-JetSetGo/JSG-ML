import random
from math import sqrt
from typing import List, Dict, Any

from fastapi import HTTPException
from domain.place.repository.place_repository import PlaceRepository
from domain.itinarity.dto.itinarity_request_dto import TabuSearchRequestDTO
from domain.itinarity.dto.itinarity_response_dto import TabuSearchResponseDTO


def preprocess_places(places: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """places 리스트를 받아, 필요한 필드를 float/int 등으로 변환."""
    for place in places:
        # x, y를 float로 변환
        place["x"] = float(place["x"])
        place["y"] = float(place["y"])

        # entry_fee가 없다면 0 할당
        if "entry_fee" not in place:
            place["entry_fee"] = 0

        # base_score 같은 커스텀 필드를 쓰고 싶다면 기본값 할당
        if "base_score" not in place:
            place["base_score"] = 10

        # 필요시 id를 정수 변환도 가능
        place["id"] = int(place["id"])
    return places


def calculate_distance(p1: tuple, p2: tuple) -> float:
    x1, y1 = p1
    x2, y2 = p2
    return sqrt((x1 - y1) ** 2 + (x2 - y2) ** 2)


def estimate_travel_cost(distance: float, transport_mode: str = 'car') -> float:
    """거리와 선호 교통수단을 기반으로 이동 비용 추정."""
    if transport_mode == 'car':
        return distance * 1000
    elif transport_mode == 'public':
        return distance * 500
    return distance * 800


def user_satisfaction_score(place_info: Dict[str, Any], user_profile: Dict[str, Any]) -> float:
    """유저 프로필(테마, must_visit 등)에 따른 만족도 스코어 계산."""
    base_score = place_info.get('base_score', 10)
    theme_score = 0

    # 테마 점수
    if 'themes' in user_profile:
        if place_info.get('category') in user_profile['themes']:
            theme_score += 5

    # must_visit 점수
    must_visit_list = user_profile.get('must_visit_list', [])
    if place_info.get('id') in must_visit_list:
        theme_score += 10

    return base_score + theme_score


def calculate_route_objective(
    route: List[int],
    places: List[Dict[str, Any]],
    user_profile: Dict[str, Any]
) -> float:
    """경로에 대한 목적함수(만족도 - 비용 - 벌점)를 계산."""
    transport = user_profile.get('preferred_transport', 'car')
    budget_amount = user_profile.get('budget_amount', float('inf'))

    total_satisfaction = 0.0
    total_cost = 0.0

    # 장소 방문 만족도와 입장료 비용
    for idx in route:
        place = places[idx]
        total_satisfaction += user_satisfaction_score(place, user_profile)
        total_cost += place.get('entry_fee', 0)

    # 이동 비용 계산
    for i in range(len(route) - 1):
        cur_idx = route[i]
        nxt_idx = route[i + 1]
        dist = calculate_distance(
            (places[cur_idx]['x'], places[cur_idx]['y']),
            (places[nxt_idx]['x'], places[nxt_idx]['y'])
        )
        total_cost += estimate_travel_cost(dist, transport)

    # 예산 초과 벌점
    penalty = 0.0
    if total_cost > budget_amount:
        penalty = (total_cost - budget_amount) * 2

    return total_satisfaction - total_cost - penalty


def generate_initial_solution(
    places: List[Dict[str, Any]],
    user_profile: Dict[str, Any],
    insert_limit: int = 5
) -> List[int]:
    """
    초기 해를 생성할 때 must_visit_list에 해당하는 장소는 무조건 포함.
    이후 일부 다른 장소를 무작위로 추가(최대 insert_limit 개).
    """
    must_visit_ids = set(user_profile.get('must_visit_list', []))
    must_indices = []
    other_indices = []

    # must_visit_list에 해당하는 장소의 인덱스 수집
    for i, p in enumerate(places):
        if p['id'] in must_visit_ids:
            must_indices.append(i)
        else:
            other_indices.append(i)

    # 무작위로 다른 장소를 추가
    random.shuffle(other_indices)
    selected_others = other_indices[:insert_limit]

    # must-visit 장소 + 선택된 다른 장소로 초기 해 구성
    init_route = must_indices + selected_others
    # 중복 제거(만약 중복이 있을 경우)
    init_route = list(dict.fromkeys(init_route))

    return init_route


def get_neighbors(
    route: List[int],
    places: List[Dict[str, Any]],
    user_profile: Dict[str, Any],
    num_neighbors: int = 5
) -> List[List[int]]:
    """
    현재 경로(route)의 이웃해를 생성하는 함수.
    - Swap(두 위치 교환)
    - Remove(단, must_visit_list 해당 위치는 제거 불가)
    - Insert(아직 사용되지 않은 장소를 경로에 삽입)
    """
    must_visit_ids = set(user_profile.get('must_visit_list', []))
    # must_visit_list에 해당하는 장소 인덱스(경로 상에서 위치)
    must_visit_positions = {
        idx for idx, pl_idx in enumerate(route)
        if places[pl_idx]['id'] in must_visit_ids
    }

    neighbors = []

    # Swap (무작위 2개 위치 교환)
    if len(route) > 1:
        for _ in range(num_neighbors):
            new_route = route[:]
            i1, i2 = random.sample(range(len(new_route)), 2)
            new_route[i1], new_route[i2] = new_route[i2], new_route[i1]
            neighbors.append(new_route)

    # Remove (must_visit 장소가 아닌 경우에만 제거)
    removable_positions = [
        i for i in range(len(route))
        if i not in must_visit_positions
    ]
    if removable_positions:
        for _ in range(num_neighbors):
            new_route = route[:]
            rm_pos = random.choice(removable_positions)
            del new_route[rm_pos]
            neighbors.append(new_route)

    # Insert (아직 사용되지 않은 장소 중 무작위로 경로에 삽입)
    used_indices = set(route)
    all_indices = set(range(len(places)))
    insert_candidates = list(all_indices - used_indices)
    if insert_candidates:
        for _ in range(num_neighbors):
            new_route = route[:]
            to_insert = random.choice(insert_candidates)
            insert_pos = random.randint(0, len(new_route))
            new_route.insert(insert_pos, to_insert)
            neighbors.append(new_route)
    return neighbors


def tabu_search_itinerary(
    places: List[Dict[str, Any]],
    user_profile: Dict[str, Any],
    max_iter: int = 500,
    tabu_tenure: int = 20
) -> Dict[str, Any]:
    """
    Tabu Search 알고리즘을 사용하여 최적 경로(동선)를 탐색.
    1) must_visit_list를 포함하는 초기 해 생성
    2) 이웃 해 탐색 시 must_visit_list가 빠지면 무시
    3) 탐색 중 Tabu List(금지 해) 관리
    """
    # 초기 해 생성
    current_route = generate_initial_solution(places, user_profile)
    current_score = calculate_route_objective(current_route, places, user_profile)

    best_route = current_route[:]
    best_score = current_score

    # 탭루 리스트(최근 해를 저장해 중복/재방문 방지)
    tabu_list = set()
    tabu_list.add(tuple(current_route))

    for _ in range(max_iter):
        neighbors = get_neighbors(current_route, places, user_profile, num_neighbors=5)

        candidate_best_route = None
        candidate_best_score = float('-inf')

        for nbr in neighbors:
            nbr_tuple = tuple(nbr)
            # Tabu 리스트에 있으면 건너뜀
            if nbr_tuple in tabu_list:
                continue

            # 경로 내 중복 장소가 있으면 건너뜀
            if len(nbr) != len(set(nbr)):
                continue

            # must_visit_list가 전부 포함되어 있는지 확인
            must_visit_ids = set(user_profile.get('must_visit_list', []))
            route_place_ids = {places[i]['id'] for i in nbr}
            if not must_visit_ids.issubset(route_place_ids):
                continue

            score = calculate_route_objective(nbr, places, user_profile)
            if score > candidate_best_score:
                candidate_best_score = score
                candidate_best_route = nbr

        if candidate_best_route is not None:
            current_route = candidate_best_route
            current_score = candidate_best_score

            if current_score > best_score:
                best_route = current_route[:]
                best_score = current_score

            tabu_list.add(tuple(current_route))
            # tabu_tenure(저장 가능 수) 초과 시 가장 오래된 해 제거
            if len(tabu_list) > tabu_tenure:
                # 집합(set)을 그대로 pop 하면 랜덤 요소가 제거되므로,
                # 순서를 보장하기 위해 리스트로 변환 & 0번 인덱스를 제거하는 식으로
                # 관리할 수도 있습니다. (여기서는 단순 예시로 pop() 사용)
                tabu_list.pop()

    best_route_place_ids = [places[i]['id'] for i in best_route]
    return {
        'best_route_place_ids': best_route_place_ids,
        'objective': best_score
    }


# ----------------------------------------------
# 여기서부터 실제 'Service' 진입점 함수
# ----------------------------------------------

def optimize_tabu_search(request: TabuSearchRequestDTO) -> TabuSearchResponseDTO:
    """
    Controller(Router)에서 호출할 실제 비즈니스 로직 함수.
    1) request 정보를 바탕으로 place_repository에서 필요한 데이터를 조회
    2) user_profile 구성
    3) tabu_search_itinerary 호출
    4) 결과를 TabuSearchResponseDTO로 감싸 반환
    """
    # 1) place_repository 통해 데이터 조회
    place_repository = PlaceRepository()
    places_detail = []
    for pid in request.places:
        places_detail.extend(place_repository.get_place_list_by_id(pid))

    user_profile_dict = request.user_profile.dict()
    # 3) Tabu Search 알고리즘 실행
    tabu_result = tabu_search_itinerary(
        places=preprocess_places(places_detail),
        user_profile=user_profile_dict,
        max_iter=request.max_iter,
        # 필요하다면 request에 tabu_tenure를 추가해서 넘길 수도 있음
    )

    # 4) 결과 DTO 변환
    response_dto = TabuSearchResponseDTO(
        best_route_place_ids=tabu_result['best_route_place_ids'],
        objective=tabu_result['objective']
    )
    return response_dto
