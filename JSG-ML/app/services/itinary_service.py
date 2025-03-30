import math
import random
import statistics
import json
from typing import List, Dict, Any, Tuple, Set, Optional


def load_places(json_path: str) -> Dict[str, Any]:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    places_dict = {}
    for item in data:
        pid = str(item["id"])
        x = float(item["x"])
        y = float(item["y"])
        review_count = item.get("reviewCount", 0)
        cat = item.get("category", [])
        if isinstance(cat, str):
            cat = [cat]
        places_dict[pid] = {
            "id": pid,
            "name": item.get("name", f"Place_{pid}"),
            "x": x,
            "y": y,
            "reviewCount": review_count,
            "category": cat,
        }
    return places_dict


def compute_distance(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    dx = a["x"] - b["x"]
    dy = a["y"] - b["y"]
    return math.sqrt(dx * dx + dy * dy)


def build_cost_matrix(places_dict: Dict[str, Any], place_ids: List[str]) -> List[List[float]]:
    n = len(place_ids)
    cost_mat = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                cost_mat[i][j] = 0.0
            else:
                p_i = places_dict[place_ids[i]]
                p_j = places_dict[place_ids[j]]
                cost_mat[i][j] = compute_distance(p_i, p_j)
    return cost_mat


def assign_prizes(
        places_dict: Dict[str, Any],
        place_ids: List[str],
        base_scale: float = 1.0,
        priority_scale: float = 0.3,
        cat_keywords: List[str] = None,
        cat_bonus: float = 10.0
) -> Tuple[Dict[str, float], Dict[int, Dict[str, float]]]:
    """
    장소별 점수(base_prize)와 방문 순서별 보너스(priority_prize) 부여
    예: 리뷰 수 * base_scale + (카테고리 매칭 시 cat_bonus), 방문 순서별 sqrt 감쇠
    """
    base_prize = {}
    priority_prize = {}

    for pid in place_ids:
        info = places_dict[pid]
        rc = info["reviewCount"]
        cat_match = 0
        if cat_keywords:
            for c in info["category"]:
                for keyword in cat_keywords:
                    if keyword in c:
                        cat_match = 1
                        break
                if cat_match:
                    break
        base_val = rc * base_scale
        if cat_match:
            base_val += cat_bonus

        base_prize[pid] = base_val

    for k in range(1, len(place_ids) + 1):
        priority_prize[k] = {}
        for pid in place_ids:
            p_base = base_prize[pid]
            priority_prize[k][pid] = p_base * priority_scale / math.sqrt(k)

    return base_prize, priority_prize


def calc_profit(
        route: List[str],
        place_ids: List[str],
        cost_matrix: List[List[float]],
        base_prize: Dict[str, float],
        priority_prize: Dict[int, Dict[str, float]],
        distance_weight: float = 1.0
) -> float:
    """
    경로 점수(방문지 점수 - 이동 거리 비용) 계산
    :param route: 방문 순서대로된 place id 리스트
    :param place_ids: 탐색 대상이 되는 모든 place id 리스트
    :param cost_matrix: i->j 이동 비용(거리) 2차원 배열
    :param base_prize: {place_id: 기본 보상} 맵
    :param priority_prize: {순서 k: {place_id: 우선순위 보너스}} 맵
    :param distance_weight: 이동 비용 가중치
    :return: route의 총 점수
    """
    if len(route) < 2:
        return 0.0

    # -------------------------------
    # 이동 비용 계산
    # -------------------------------
    total_cost = 0.0
    for i in range(len(route) - 1):
        idx_i = place_ids.index(route[i])
        idx_j = place_ids.index(route[i + 1])
        total_cost += cost_matrix[idx_i][idx_j]

    # -------------------------------
    # (※ 중요) 방문 보상 + 우선순위 보상 계산
    # -------------------------------
    total_prize = 0.0
    for k, pid in enumerate(route):
        order = k + 1
        # ---- (수정 필요) 방문하면 무조건 base_prize 반영
        total_prize += base_prize.get(pid, 0.0)

        # ---- (수정 필요) 우선순위 만족 시 추가로 priority_prize
        if order in priority_prize:
            # priority_prize[order]는 {place_id: val} 형태
            total_prize += priority_prize[order].get(pid, 0.0)

    # -------------------------------
    # 최종 계산: (방문 보상 합계 + 우선순위 보너스) - 이동 비용
    # -------------------------------
    return total_prize - distance_weight * total_cost


# ----------------------------------------------------------
# Tabu Search 관련 함수

def two_opt(route: List[str], fixed_count: int) -> List[str]:
    if len(route) <= fixed_count + 2:
        return route[:]
    var_start = fixed_count + 1
    var_end = len(route) - 2
    if var_start >= var_end:
        return route[:]
    i = random.randint(var_start, var_end - 1)
    j = random.randint(i + 1, var_end)
    new_route = route[:]
    new_route[i:j + 1] = reversed(new_route[i:j + 1])
    return new_route


def pairwise_swap(route: List[str], fixed_count: int) -> List[str]:
    if len(route) <= fixed_count + 2:
        return route[:]
    var_idx = list(range(fixed_count + 1, len(route) - 1))
    if len(var_idx) < 2:
        return route[:]
    i, j = random.sample(var_idx, 2)
    new_route = route[:]
    new_route[i], new_route[j] = new_route[j], new_route[i]
    return new_route


def remove_node(route: List[str], fixed_count: int) -> Tuple[List[str], Optional[str]]:
    if len(route) <= fixed_count + 2:
        return route[:], None
    var_idx = list(range(fixed_count + 1, len(route) - 1))
    rm_idx = random.choice(var_idx)
    removed_node = route[rm_idx]
    new_route = route[:rm_idx] + route[rm_idx + 1:]
    return new_route, removed_node


def reinsert_node(route: List[str], unvisited: List[str], fixed_count: int) -> List[str]:
    inside = set(route[fixed_count + 1:-1])
    candidates = [n for n in unvisited if n not in inside]
    if not candidates:
        return route[:]
    node = random.choice(candidates)
    pos = random.randint(fixed_count + 1, len(route) - 1)
    new_route = route[:]
    new_route.insert(pos, node)
    return new_route


def get_neighbors(route: List[str],
                  unvisited: List[str],
                  fixed_count: int) -> Tuple[List[List[str]], Optional[str]]:
    nbrs = []
    t2 = two_opt(route, fixed_count)
    pswap = pairwise_swap(route, fixed_count)
    rnode, removed = remove_node(route, fixed_count)
    rinsert = reinsert_node(route, unvisited, fixed_count)
    nbrs.extend([t2, pswap, rnode, rinsert])
    return nbrs, removed


def check_daily_constraints(
        route: List[str],
        valid_pids: List[str],
        cost_matrix: List[List[float]],
        num_days: int,
        daily_limit: int,
        daily_max_distance: float,
        daily_max_duration: float
) -> bool:
    """
    route (start->...->start)를
    '연속형'으로 day1, day2, ... dayN을 순차 분할하되,
    하루에 최대 daily_limit 개의 장소를 배정.

    각 일자의 이동거리와 소요시간이 daily_max_* 이하인지 검사.
    초과 시 False, 모두 만족하면 True.
    """
    if num_days < 1 or daily_limit < 1:
        return True  # 혹은 False, 상황에 맞게

    # route[0], route[-1] = 동일 노드
    inner = route[1:-1]
    route_len = len(inner)

    curr_index = 0
    current_start = route[0]

    for d in range(num_days):
        # 하루에 최대 daily_limit 방문
        next_idx = curr_index + daily_limit
        if next_idx > route_len:
            next_idx = route_len

        today_inner = inner[curr_index:next_idx]
        curr_index = next_idx

        day_route = [current_start] + today_inner

        if d == num_days - 1:
            # 마지막 날이면 나머지 전부 + route[-1]
            day_route.append(route[-1])
        else:
            # 오늘 마지막 방문지가 내일 시작
            if today_inner:
                current_start = today_inner[-1]
            else:
                # 방문지가 없는 경우 -> start_node 그대로
                current_start = current_start

        # 이동거리/시간 계산
        dist = 0.0
        for i in range(len(day_route) - 1):
            idx1 = valid_pids.index(day_route[i])
            idx2 = valid_pids.index(day_route[i + 1])
            dist += cost_matrix[idx1][idx2]

        # 예: 1km당 0.15h => 9분
        travel_time = dist * 0.15
        stay_time = (len(day_route) - 1) * 1.0
        day_duration = travel_time + stay_time

        # 제약 검사
        if dist > daily_max_distance:
            return False
        if day_duration > daily_max_duration:
            return False

    return True


def is_valid_route(
        route: List[str],
        start_node: str,
        must_visit: List[str],
        max_visit: int,  # 내부 방문지 최대치
        daily_start_points: List[int],
        num_days: int,
        daily_limit: int,
        daily_max_distance: float,
        daily_max_duration: float,
        valid_pids: List[str],
        cost_matrix: List[List[float]]
) -> bool:
    """
    1) 경로 시작/끝 start_node 동일
    2) 내부 방문지 <= max_visit
    3) must_visit 모두 포함
    4) 앵커 순서
    5) 일자별 daily_limit, daily_max_distance/duration 충족(check_daily_constraints)
    """
    if len(route) < 2:
        return False
    if route[0] != start_node or route[-1] != start_node:
        return False

    inner = route[1:-1]
    # 중복 방문X
    if len(inner) != len(set(inner)):
        return False
    # 최대 방문 제한
    if len(inner) > max_visit:
        return False

    # 필수 방문지 포함
    for mv in must_visit:
        if mv not in inner:
            return False

    # 앵커 검증(기존 로직)
    if daily_start_points:
        anchor_ids = [str(x) for x in daily_start_points if x is not None]
        if len(anchor_ids) > 1:
            positions = {}
            for anc in anchor_ids:
                try:
                    positions[anc] = route.index(anc)
                except ValueError:
                    return False
            # 첫 앵커는 route[0]이어야 함
            if positions.get(anchor_ids[0], None) != 0:
                return False
            # 순서 체크
            for j in range(1, len(anchor_ids)):
                prev_anchor = anchor_ids[j - 1]
                curr_anchor = anchor_ids[j]
                if positions[prev_anchor] >= positions[curr_anchor]:
                    return False

    # 마지막: 일자별 거리/시간 제약
    if not check_daily_constraints(
            route, valid_pids, cost_matrix,
            num_days, daily_limit,
            daily_max_distance, daily_max_duration
    ):
        return False

    return True


def advanced_tabu_search(
        place_ids: List[str],
        cost_matrix: List[List[float]],
        base_prize: Dict[str, float],
        priority_prize: Dict[int, Dict[str, float]],
        start_node: str,
        must_visit: List[str],
        max_visit: int,  # 전체 내부 방문지 최대치
        daily_start_points: List[int],
        num_days: int,
        daily_limit: int,
        daily_max_distance: float,
        daily_max_duration: float,
        valid_pids: List[str],
        max_iter: int = 50,
        hmin: int = 5,
        hmax: int = 10
) -> Tuple[List[str], float]:
    """
    Tabu Search 함수:
      - is_valid_route(...) 내에서 check_daily_constraints(...) 호출하여
        일자별 거리/시간 제약을 검사
      - calc_profit(...)에서 priority_prize를 실제 반영
    """

    # 1) must_visit 정리
    seen = set()
    must_visit_list = []

    for m in must_visit:
        if m != start_node and m not in seen:
            must_visit_list.append(m)
            seen.add(m)

    # 필수 방문지가 max_visit보다 많으면 불가능
    if len(must_visit_list) > max_visit:
        raise ValueError("필수 방문지 > max_visit")

    # 나머지 후보
    others = [p for p in place_ids if p not in set(must_visit_list) and p != start_node]

    # 2) 초기 해 구성
    init_route = None
    attempts = 100
    while attempts > 0 and init_route is None:
        route_candidate = [start_node] + must_visit_list[:]

        # 0 ~ (max_visit - len(must_visit_list))개 추가
        can_add = max_visit - len(must_visit_list)
        if can_add > 0 and len(others) > 0:
            add_cnt = random.randint(0, min(can_add, len(others)))
            add_nodes = random.sample(others, add_cnt)
        else:
            add_nodes = []

        route_candidate += add_nodes
        route_candidate.append(start_node)

        # 유효성 검사
        if is_valid_route(
                route_candidate,
                start_node,
                must_visit_list,
                max_visit,
                daily_start_points,
                num_days,
                daily_limit,
                daily_max_distance,
                daily_max_duration,
                valid_pids,
                cost_matrix
        ):
            init_route = route_candidate
        attempts -= 1

    if init_route is None:
        raise RuntimeError("초기 해 구성 실패 (제약 과도)")

    # 3) Tabu Search 시작
    current_route = init_route[:]
    best_route = current_route[:]
    best_score = -float("inf")

    # calc_profit에서 priority_prize 사용 (방문 순서별 보너스)
    def calc_profit(route: List[str]) -> float:
        distance_weight = 1.0  # 필요 시 가중치 조정 가능
        # 이동 거리 합
        total_cost = 0.0
        for i in range(len(route) - 1):
            idx_i = place_ids.index(route[i])
            idx_j = place_ids.index(route[i + 1])
            total_cost += cost_matrix[idx_i][idx_j]

        # 방문지 점수 + 우선순위 보너스
        total_prize = 0.0
        for k, pid in enumerate(route):
            order = k + 1
            # 기본 점수
            total_prize += base_prize.get(pid, 0.0)
            # priority_prize가 존재하면 추가
            if order in priority_prize:
                total_prize += priority_prize[order].get(pid, 0.0)

        return total_prize - distance_weight * total_cost

    best_score = calc_profit(best_route)

    fixed_count = len(must_visit_list)
    tabu_list = []
    TABU_CAPA = 8

    stage_length = random.randint(hmin, hmax)
    stage_iter_count = 0
    iteration = 0
    stage_scores: List[float] = []
    prev_stage_mean = None
    unvisited: Set[str] = set()

    while iteration < max_iter:
        iteration += 1
        stage_iter_count += 1

        # 이웃 생성
        neighbors, removed_node = get_neighbors(current_route, list(unvisited), fixed_count)
        best_neighbor_route = None
        best_neighbor_score = -float("inf")

        for nbr in neighbors:
            # 유효성 (일자별 제약 등)
            if not is_valid_route(
                    nbr,
                    start_node,
                    must_visit_list,
                    max_visit,
                    daily_start_points,
                    num_days,
                    daily_limit,
                    daily_max_distance,
                    daily_max_duration,
                    valid_pids,
                    cost_matrix
            ):
                continue

            route_key = tuple(nbr)
            sc_nbr = calc_profit(nbr)

            # 금기(tabu) 검사
            if route_key in tabu_list and sc_nbr <= best_score:
                continue

            # 더 나은 후보인지 갱신
            if sc_nbr > best_neighbor_score:
                best_neighbor_score = sc_nbr
                best_neighbor_route = nbr

        # 이웃 중 valid를 못 찾으면 fallback
        if best_neighbor_route is None:
            fallback_route = two_opt(current_route, fixed_count)
            if is_valid_route(
                    fallback_route,
                    start_node,
                    must_visit_list,
                    max_visit,
                    daily_start_points,
                    num_days,
                    daily_limit,
                    daily_max_distance,
                    daily_max_duration,
                    valid_pids,
                    cost_matrix
            ):
                best_neighbor_route = fallback_route
                best_neighbor_score = calc_profit(fallback_route)
            else:
                # 더 이상 진행 불가 -> 중단
                break

        current_route = best_neighbor_route[:]
        tabu_list.append(tuple(current_route))
        if len(tabu_list) > TABU_CAPA:
            tabu_list.pop(0)

        if best_neighbor_score > best_score:
            best_score = best_neighbor_score
            best_route = best_neighbor_route[:]

        stage_scores.append(best_neighbor_score)
        if stage_iter_count >= stage_length:
            curr_mean = statistics.mean(stage_scores)
            # 정체(stagnation) 판단
            if prev_stage_mean is not None and (curr_mean - prev_stage_mean) < 1e-6:
                tabu_list.clear()
            stage_iter_count = 0
            stage_length = random.randint(hmin, hmax)
            stage_scores = []
            prev_stage_mean = curr_mean

    return best_route, best_score


def post_process_daily_route_continuous(
        best_overall_route: List[str],
        num_days: int,
        valid_pids: List[str],
        cost_matrix: List[List[float]],
        best_daily_limit: int,
        daily_max_distance: Optional[float] = 50,
        daily_max_duration: Optional[float] = 100
) -> Tuple[List[Dict[str, Any]], float]:
    overall_distance = 0.0
    for i in range(len(best_overall_route) - 1):
        idx_i = valid_pids.index(best_overall_route[i])
        idx_j = valid_pids.index(best_overall_route[i + 1])
        overall_distance += cost_matrix[idx_i][idx_j]

    inner_route = best_overall_route[1:-1]
    route_len = len(inner_route)

    daily_itineraries = []
    curr_index = 0
    current_start_node = best_overall_route[0]

    for day in range(num_days):
        real_end_index = curr_index + best_daily_limit
        if real_end_index > route_len:
            real_end_index = route_len

        day_inner_part = inner_route[curr_index:real_end_index]
        curr_index = real_end_index

        day_route = [current_start_node] + day_inner_part

        if day == num_days - 1:
            day_route.append(best_overall_route[-1])
        else:
            if day_inner_part:
                next_start_node = day_inner_part[-1]
            else:
                next_start_node = current_start_node
            current_start_node = next_start_node

        daily_distance = 0.0
        for k in range(len(day_route) - 1):
            idx1 = valid_pids.index(day_route[k])
            idx2 = valid_pids.index(day_route[k + 1])
            daily_distance += cost_matrix[idx1][idx2]

        travel_time = daily_distance * 0.15
        visit_time = (len(day_route) - 1) * 1.0
        daily_duration = travel_time + visit_time

        penalty = 0.0
        if daily_max_distance is not None and daily_distance > daily_max_distance:
            penalty += (daily_distance - daily_max_distance) * 10.0
        if daily_max_duration is not None and daily_duration > daily_max_duration:
            penalty += (daily_duration - daily_max_duration) * 10.0

        daily_duration_with_penalty = daily_duration + penalty

        daily_itineraries.append({
            "day": day + 1,
            "route": [int(x) for x in day_route],
            "daily_distance": round(daily_distance, 2),
            "daily_duration": round(daily_duration_with_penalty, 2),
            "used_daily_limit": best_daily_limit
        })

    return daily_itineraries, overall_distance


def calculate_itinerary(request_data: Dict[str, Any],
                        places_json_dir: str = "./app/data/") -> Tuple[List[Dict[str, Any]], float]:
    """
    - 방문하기 싫은 장소(not_visit_list) 제외
    - must_visit_list + anchor 반영
    - 하루 max_places_per_day <= m
    - 각 일자 방문지는 m개 이하 (즉, 전체 방문 <= m*num_days)
    - Tabu Search로 경로 생성 후, 일자별 연속형 분할
    """
    # 1) load places
    places_field = request_data.get("places")
    if places_field is None:
        raise ValueError("places 필드가 필요합니다.")
    if isinstance(places_field, int):
        db_nums = [places_field]
    elif isinstance(places_field, list):
        db_nums = places_field
    else:
        raise ValueError("places 필드는 정수 또는 정수 리스트여야 합니다.")

    places_dict_total: Dict[str, Any] = {}
    for db_num in db_nums:
        json_path = f"{places_json_dir}{db_num}_관광지.json"
        partial_places = load_places(json_path)
        places_dict_total.update(partial_places)

    if not places_dict_total:
        raise ValueError("로드된 장소 데이터가 없습니다.")

    user_profile = request_data.get("user_profile", {})
    themes = user_profile.get("themes", [])
    must_visit_input = [str(x) for x in user_profile.get("must_visit_list", [])]
    not_visit_input = [str(x) for x in user_profile.get("not_visit_list", [])]

    # not_visit_list 반영
    valid_pids_all = list(places_dict_total.keys())
    valid_pids = [pid for pid in valid_pids_all if pid not in not_visit_input]

    # must_visit과 not_visit 겹치면 에러
    overlap = set(must_visit_input).intersection(set(not_visit_input))
    if overlap:
        raise ValueError(f"must_visit과 not_visit이 겹칩니다: {overlap}")

    # build cost matrix
    cost_matrix = build_cost_matrix(places_dict_total, valid_pids)

    num_days = request_data.get("num_days")
    max_places_per_day = request_data.get("max_places_per_day")
    daily_start_points_input = request_data.get("daily_start_points", [])

    if num_days is None or max_places_per_day is None or not isinstance(daily_start_points_input, list):
        raise ValueError("num_days, max_places_per_day, daily_start_points가 필요합니다.")

    # anchor
    if len(daily_start_points_input) < num_days:
        daily_start_points_input += [None] * (num_days - len(daily_start_points_input))
    else:
        daily_start_points_input = daily_start_points_input[:num_days]

    if daily_start_points_input[0] is None:
        overall_start = valid_pids[0]
    else:
        overall_start = str(daily_start_points_input[0])

    # must_visit + anchor
    additional_anchors = [str(x) for x in daily_start_points_input[1:] if x is not None]
    combined_must_visit = must_visit_input[:]
    for anc in additional_anchors:
        if anc not in combined_must_visit:
            combined_must_visit.append(anc)

    # assign prize
    base_prz, prio_prz = assign_prizes(
        places_dict_total, valid_pids,
        base_scale=0.1, priority_scale=2,
        cat_keywords=themes, cat_bonus=100.0
    )

    best_overall_route = None
    best_overall_score = -float("inf")
    best_daily_limit = None

    # daily_limit from 1 to max_places_per_day
    for daily_limit in range(1, max_places_per_day + 1):
        # max total: daily_limit * num_days
        max_total_visit = daily_limit * num_days
        try:
            route_candidate, score_candidate = advanced_tabu_search(
                place_ids=valid_pids,
                cost_matrix=cost_matrix,
                base_prize=base_prz,
                priority_prize=prio_prz,
                start_node=overall_start,
                must_visit=combined_must_visit,
                max_visit=max_total_visit,  # 내부 방문지 최대치
                daily_start_points=daily_start_points_input,
                num_days=num_days,  # 추가: 요청 데이터에서 가져온 num_days
                daily_limit=max_places_per_day,  # 추가: 하루 최대 방문지 수
                daily_max_distance=request_data.get("daily_max_distance", 5.0),  # 추가: 요청 데이터의 일일 최대 거리
                daily_max_duration=request_data.get("daily_max_duration", 10.0),  # 추가: 요청 데이터의 일일 최대 시간
                valid_pids=valid_pids,  # 추가: 유효한 place id 리스트
                max_iter=request_data.get("max_iter", 50)
            )
            import logging
            logging.log(logging.DEBUG, f"{route_candidate}, {score_candidate}")

            if score_candidate > best_overall_score:
                best_overall_score = score_candidate
                best_overall_route = route_candidate
                best_daily_limit = daily_limit
        except Exception as e:
            import logging
            logging.log(logging.DEBUG, f"{e}")

            continue

    if best_overall_route is None:
        raise RuntimeError("전체 최적 해를 찾지 못했습니다.")

    # 일자별로 분할 (연속형)
    daily_max_distance = request_data.get("daily_max_distance", None)
    daily_max_duration = request_data.get("daily_max_duration", None)

    daily_itineraries, overall_distance = post_process_daily_route_continuous(
        best_overall_route,
        num_days,
        valid_pids,
        cost_matrix,
        best_daily_limit,
        daily_max_distance,
        daily_max_duration
    )

    return daily_itineraries, overall_distance
