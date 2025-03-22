import math
import random
import statistics
import json
from typing import List, Dict, Any, Tuple, Set


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


def assign_prizes(places_dict: Dict[str, Any],
                  place_ids: List[str],
                  base_scale: float = 1.0,
                  priority_scale: float = 0.3,
                  cat_keywords: List[str] = None,
                  cat_bonus: float = 10.0) -> Tuple[Dict[str, float], Dict[int, Dict[str, float]]]:
    base_prize = {}
    n = len(place_ids)
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

    for k in range(1, n + 1):
        priority_prize[k] = {}
        for pid in place_ids:
            p_base = base_prize[pid]
            priority_prize[k][pid] = p_base * priority_scale / math.sqrt(k)
    return base_prize, priority_prize


def calc_profit(route: List[str],
                place_ids: List[str],
                cost_matrix: List[List[float]],
                base_prize: Dict[str, float],
                priority_prize: Dict[int, Dict[str, float]],
                distance_weight: float = 1.0) -> float:
    if len(route) < 2:
        return 0.0
    total_cost = 0.0
    for i in range(len(route) - 1):
        idx_i = place_ids.index(route[i])
        idx_j = place_ids.index(route[i + 1])
        total_cost += cost_matrix[idx_i][idx_j]
    total_prize = 0.0
    for k, pid in enumerate(route):
        order = k + 1
        total_prize += base_prize.get(pid, 0.0)
        if order in priority_prize:
            total_prize += priority_prize[order].get(pid, 0.0)
    return total_prize - distance_weight * total_cost


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


def remove_node(route: List[str], fixed_count: int) -> Tuple[List[str], Any]:
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
                  fixed_count: int) -> Tuple[List[List[str]], Any]:
    nbrs = []
    t2 = two_opt(route, fixed_count)
    pswap = pairwise_swap(route, fixed_count)
    rnode, removed = remove_node(route, fixed_count)
    rinsert = reinsert_node(route, unvisited, fixed_count)
    nbrs.extend([t2, pswap, rnode, rinsert])
    return nbrs, removed


def is_valid_route(route: List[str],
                   start_node: str,
                   must_visit: List[str],
                   exact_visit: int) -> bool:
    if len(route) < 2:
        return False
    if route[0] != start_node or route[-1] != start_node:
        return False
    inner = route[1:-1]

    if len(inner) != len(set(inner)):
        return False
    if len(inner) != exact_visit:
        return False
    if start_node in inner:
        return False
    for m in must_visit:
        if m not in inner:
            return False
    return True


def advanced_tabu_search(place_ids: List[str],
                         cost_matrix: List[List[float]],
                         base_prize: Dict[str, float],
                         priority_prize: Dict[int, Dict[str, float]],
                         start_node: str,
                         must_visit: List[str],
                         exact_visit: int,
                         max_iter: int = 50,
                         hmin: int = 5, hmax: int = 10) -> Tuple[List[str], float]:
    others = [p for p in place_ids if p not in set(must_visit) | {start_node}]
    if exact_visit < len(must_visit):
        raise ValueError("must_visit 개수가 요구 방문지 수보다 많습니다.")
    attempts = 100
    found_init = False
    init_route = None
    while attempts > 0 and not found_init:
        if len(others) >= (exact_visit - len(must_visit)):
            add_nodes = random.sample(others, exact_visit - len(must_visit))
        else:
            add_nodes = others[:]
        route = [start_node] + must_visit + add_nodes + [start_node]
        if is_valid_route(route, start_node, must_visit, exact_visit):
            found_init = True
            init_route = route
        else:
            attempts -= 1
    if not found_init or not init_route:
        raise RuntimeError("초기 해 구성에 실패했습니다.")
    current_route = init_route[:]
    best_route = current_route[:]
    best_score = calc_profit(best_route, place_ids, cost_matrix, base_prize, priority_prize)
    fixed_count = len(must_visit)
    tabu_list = []
    TABU_CAPA = 8
    stage_length = random.randint(hmin, hmax)
    stage_iter_count = 0
    iteration = 0
    stage_scores = []
    prev_stage_mean = None
    unvisited: Set[str] = set()
    while iteration < max_iter:
        iteration += 1
        stage_iter_count += 1
        neighbors, removed_node = get_neighbors(current_route, list(unvisited), fixed_count)
        best_n_sol = None
        best_n_score = float("-inf")
        for nr in neighbors:
            if not is_valid_route(nr, start_node, must_visit, exact_visit):
                continue
            key = tuple(nr)
            sc = calc_profit(nr, place_ids, cost_matrix, base_prize, priority_prize)
            if key in tabu_list and sc <= best_score:
                continue
            if sc > best_n_score:
                best_n_score = sc
                best_n_sol = nr
        if best_n_sol is None:
            fallback = two_opt(current_route, fixed_count)
            if is_valid_route(fallback, start_node, must_visit, exact_visit):
                best_n_sol = fallback
                best_n_score = calc_profit(best_n_sol, place_ids, cost_matrix, base_prize, priority_prize)
            else:
                best_n_sol = current_route[:]
                best_n_score = calc_profit(current_route, place_ids, cost_matrix, base_prize, priority_prize)
        old_inner = set(current_route[1:-1])
        new_inner = set(best_n_sol[1:-1])
        removed_set = old_inner - new_inner
        if removed_set:
            unvisited.update(removed_set)
        current_route = best_n_sol
        current_score = best_n_score
        if current_score > best_score:
            best_score = current_score
            best_route = current_route[:]
        tabu_list.append(tuple(current_route))
        if len(tabu_list) > TABU_CAPA:
            tabu_list.pop(0)
        stage_scores.append(current_score)
        if stage_iter_count >= stage_length:
            curr_mean = sum(stage_scores) / len(stage_scores)
            stage_length = random.randint(hmin, hmax)
            stage_iter_count = 0
            stage_scores = []
            prev_stage_mean = curr_mean

    return best_route, best_score


def calculate_itinerary(request_data: Dict[str, Any],
                        places_json_dir: str = "./app/data/") -> Tuple[List[Dict[str, Any]], float]:
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
        places = load_places(json_path)
        places_dict_total.update(places)
    if not places_dict_total:
        raise ValueError("로드된 장소 데이터가 없습니다.")
    valid_pids = list(places_dict_total.keys())

    cost_matrix = build_cost_matrix(places_dict_total, valid_pids)
    user_profile = request_data.get("user_profile", {})
    themes = user_profile.get("themes", [])
    must_visit = [str(x) for x in user_profile.get("must_visit_list", [])]
    base_prz, prio_prz = assign_prizes(
        places_dict_total,
        valid_pids,
        base_scale=0.1,
        priority_scale=2,
        cat_keywords=themes,
        cat_bonus=100.0
    )
    max_iter = request_data.get("max_iter", 50)

    num_days = request_data.get("num_days")
    max_places_per_day = request_data.get("max_places_per_day")
    daily_start_points = request_data.get("daily_start_points")
    if num_days is None or max_places_per_day is None or daily_start_points is None:
        raise ValueError("num_days, max_places_per_day, 그리고 daily_start_points가 필요합니다.")
    if len(daily_start_points) != num_days:
        raise ValueError("daily_start_points의 개수는 num_days와 일치해야 합니다.")

    best_overall_route = None
    best_overall_score = -float("inf")
    best_daily_limit = None
    overall_start = str(daily_start_points[0])

    for daily_limit in range(1, max_places_per_day + 1):
        total_visits = daily_limit * num_days
        try:
            route_candidate, score_candidate = advanced_tabu_search(
                place_ids=valid_pids,
                cost_matrix=cost_matrix,
                base_prize=base_prz,
                priority_prize=prio_prz,
                start_node=overall_start,
                must_visit=must_visit,
                exact_visit=total_visits,
                max_iter=max_iter
            )
            if score_candidate > best_overall_score:
                best_overall_score = score_candidate
                best_overall_route = route_candidate
                best_daily_limit = daily_limit
        except Exception as e:
            continue
    if best_overall_route is None:
        raise RuntimeError("전체 최적 해를 찾지 못했습니다.")

    overall_route = best_overall_route[:-1]
    if len(overall_route) != (best_daily_limit * num_days + 1):
        raise RuntimeError("전체 경로의 내부 방문지 수가 기대와 다릅니다.")

    overall_route = best_overall_route[:-1]
    overall_start_node = overall_route[0]
    inner_route = overall_route[1:]
    if len(inner_route) != best_daily_limit * num_days:
        raise RuntimeError("내부 방문지 수가 기대와 다릅니다.")

    daily_itineraries = []
    overall_distance = 0.0

    for i in range(len(best_overall_route) - 1):
        idx_i = valid_pids.index(best_overall_route[i])
        idx_j = valid_pids.index(best_overall_route[i + 1])
        overall_distance += cost_matrix[idx_i][idx_j]

    for day in range(num_days):
        start_idx = day * best_daily_limit
        end_idx = (day + 1) * best_daily_limit
        segment = inner_route[start_idx:end_idx]
        if day == 0:
            day_route = [overall_start_node] + segment
        elif day == num_days - 1:
            day_route = segment + [overall_start_node]
        else:
            day_route = segment

        daily_distance = 0.0
        if len(day_route) > 1:
            for j in range(len(day_route) - 1):
                idx_i = valid_pids.index(day_route[j])
                idx_j = valid_pids.index(day_route[j + 1])
                daily_distance += cost_matrix[idx_i][idx_j]
        daily_itineraries.append({
            "day": day + 1,
            "route": [int(x) for x in day_route],
            "daily_distance": daily_distance,
            "used_daily_limit": best_daily_limit
        })

    return daily_itineraries, overall_distance
