import math
import random
import json
from typing import List, Dict, Any, Tuple

import requests
from concurrent.futures import ThreadPoolExecutor
from sklearn.cluster import KMeans

import pulp

# NEW: Database-related imports
import os
import pymysql
from dotenv import load_dotenv

load_dotenv()  # Make sure your .env file is in the same folder or properly referenced

DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")


def load_places_from_db(city_id: int) -> Dict[str, Any]:
    """
    DB의 tourist_spots 테이블에서 travel_city_id에 해당하는 데이터를 불러와,
    기존 JSON 파일 구조에 맞게 places_dict를 생성합니다.
    사용 컬럼:
      - tourist_spot_id → id
      - activity_level → reviewCount (대용)
      - latitude, longitude → y, x
      - name → name
      - category → category (JSON 파싱 시도)
    """
    connection = pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        charset='utf8'
    )
    places_dict = {}

    try:
        with connection.cursor() as cursor:
            sql = """
                SELECT 
                    tourist_spot_id,
                    activity_level,
                    latitude,
                    longitude,
                    name,
                    category
                FROM tourist_spots
                WHERE travel_city_id = %s
            """
            cursor.execute(sql, (city_id,))
            rows = cursor.fetchall()

            for row in rows:
                # row 순서: (tourist_spot_id, activity_level, latitude, longitude, name, category)
                pid = str(row[0])
                # activity_level을 reviewCount로 사용 (없으면 0)
                try:
                    review_count = float(row[1]) if row[1] is not None else 0.0
                except Exception:
                    review_count = 0.0

                # 위도/경도: longitude를 x, latitude를 y로 사용
                try:
                    x_val = float(row[3]) if row[3] is not None else 0.0
                    y_val = float(row[2]) if row[2] is not None else 0.0
                except Exception:
                    x_val, y_val = 0.0, 0.0

                # 이름 처리: 없으면 기본값
                name_val = row[4] if row[4] else f"Place_{pid}"

                # category 컬럼: JSON 문자열 또는 일반 문자열일 수 있음
                cat_data = row[5]
                if cat_data:
                    try:
                        cat = json.loads(cat_data)
                        if isinstance(cat, str):
                            cat = [cat]
                        elif isinstance(cat, dict):
                            cat = [cat]
                    except Exception:
                        # JSON 파싱 실패 시, 단순 문자열로 처리
                        cat = [cat_data]
                else:
                    cat = []

                places_dict[pid] = {
                    "id": pid,
                    "name": name_val,
                    "x": x_val,
                    "y": y_val,
                    "reviewCount": review_count,
                    "category": cat,
                }
    finally:
        connection.close()

    return places_dict


def compute_distance(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    lon1, lat1 = a["x"], a["y"]
    lon2, lat2 = b["x"], b["y"]
    distance = None
    try:
        url = f"http://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?overview=false"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("routes"):
                distance_m = data["routes"][0]["distance"]
                distance = distance_m / 1000.0  # meters to kilometers
    except Exception:
        distance = None
    if distance is None:
        rad = math.pi / 180.0
        dlat = (lat2 - lat1) * rad
        dlon = (lon2 - lon1) * rad
        a_c = math.sin(dlat / 2) ** 2 + math.cos(lat1 * rad) * math.cos(lat2 * rad) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a_c), math.sqrt(1 - a_c))
        distance = 6371.0 * c
    return distance


def build_cost_matrix(places_dict: Dict[str, Any], place_ids: List[str]) -> List[List[float]]:
    n = len(place_ids)
    cost_mat = [[0.0] * n for _ in range(n)]

    from concurrent.futures import ThreadPoolExecutor

    def compute_pair(i: int, j: int) -> Tuple[Tuple[int, int], float]:
        if i == j:
            return (i, j), 0.0
        p_i = places_dict[place_ids[i]]
        p_j = places_dict[place_ids[j]]
        return (i, j), compute_distance(p_i, p_j)

    # 모든 (i, j) 쌍에 대해 계산을 병렬로 수행
    pairs = [(i, j) for i in range(n) for j in range(n)]
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda ij: compute_pair(*ij), pairs))
    for (i, j), dist in results:
        cost_mat[i][j] = dist

    return cost_mat

def assign_prizes(
        places_dict: Dict[str, Any],
        place_ids: List[str],
        base_scale: float = 1.0,
        priority_scale: float = 0.3,
        cat_keywords: List[str] = None,
        cat_bonus: float = 10.0
) -> Tuple[Dict[str, float], Dict[int, Dict[str, float]]]:
    base_prize: Dict[str, float] = {}
    priority_prize: Dict[int, Dict[str, float]] = {}

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
            p_base = base_prize.get(pid, 0.0)
            # Example formula: p_ki = p_base * priority_scale / sqrt(k)
            priority_prize[k][pid] = p_base * priority_scale / math.sqrt(k)
    return base_prize, priority_prize


def solve_day_ptppp_milp(
        places_dict: Dict[str, Any],
        day_places: List[str],
        start_pid: str,
        daily_max_distance: float,
        daily_max_duration: float,
        max_places_per_day: int,
        base_prz: Dict[str, float],
        prio_prz: Dict[int, Dict[str, float]],
        must_visit_ids: List[str],
        transport_speed_kmh: float = 40.0,
) -> Tuple[List[str], float, float]:
    if not day_places:
        return [], 0.0, 0.0

    # Remove duplicates
    day_places = list(set(day_places))
    if start_pid not in day_places:
        day_places.append(start_pid)
    if len(day_places) > max_places_per_day:
        # Exclude start_pid, sort the rest by base_prz, and keep top
        temp = [pid for pid in day_places if pid != start_pid]
        temp.sort(key=lambda x: base_prz.get(x, 0.0), reverse=True)
        chosen = temp[: (max_places_per_day - 1)]
        day_places = [start_pid] + chosen

    place_ids = [start_pid] + [pid for pid in day_places if pid != start_pid]
    n = len(place_ids) - 1
    K = min(n, max_places_per_day - 1)

    idx_to_pid = {0: start_pid}
    for i, pid in enumerate(place_ids[1:], start=1):
        idx_to_pid[i] = pid
    pid_to_idx = {v: k for k, v in idx_to_pid.items()}
    coords_list = place_ids
    cost_mat = build_cost_matrix(places_dict, coords_list)

    prob = pulp.LpProblem(f"PTPPP_Day", pulp.LpMaximize)

    X = pulp.LpVariable.dicts(
        "X",
        [(i, j) for i in range(len(place_ids)) for j in range(len(place_ids))],
        cat=pulp.LpBinary
    )
    Y = pulp.LpVariable.dicts(
        "Y",
        [(k, i) for k in range(1, K + 1) for i in range(1, n + 1)],
        cat=pulp.LpBinary
    )
    Z = pulp.LpVariable.dicts(
        "Z",
        [i for i in range(1, n + 1)],
        lowBound=0,
        cat=pulp.LpContinuous
    )

    # Objective: Maximize prize - cost
    prize_terms = []
    for i in range(1, n + 1):
        pid = idx_to_pid[i]
        base_val = base_prz.get(pid, 0.0)
        base_expr = base_val * pulp.lpSum([Y[k, i] for k in range(1, K + 1)])
        prize_terms.append(base_expr)
        for k in range(1, K + 1):
            prio_val = prio_prz.get(k, {}).get(pid, 0.0)
            prize_terms.append(prio_val * Y[k, i])

    cost_terms = []
    for i in range(len(place_ids)):
        for j in range(len(place_ids)):
            if i == j:
                continue
            cost_terms.append(cost_mat[i][j] * X[(i, j)])

    prob += pulp.lpSum(prize_terms) - pulp.lpSum(cost_terms), "Max_Prize_minus_Cost"

    # Constraints
    prob += pulp.lpSum([X[(0, j)] for j in range(1, n + 1)]) <= 1, "Start_leaving"
    prob += pulp.lpSum([X[(i, 0)] for i in range(1, n + 1)]) <= 1, "End_return"

    for i in range(1, n + 1):
        inbound = pulp.lpSum([X[(h, i)] for h in range(n + 1) if h != i])
        outbound = pulp.lpSum([X[(i, h)] for h in range(n + 1) if h != i])
        visited = pulp.lpSum([Y[(k, i)] for k in range(1, K + 1)])
        prob += inbound == visited, f"InboundNode_{i}"
        prob += outbound == visited, f"OutboundNode_{i}"

    for i in range(1, n + 1):
        prob += pulp.lpSum([Y[(k, i)] for k in range(1, K + 1)]) <= 1, f"OneOrder_{i}"

    for k in range(1, K + 1):
        prob += pulp.lpSum([Y[(k, i)] for i in range(1, n + 1)]) <= 1, f"OrderCap_{k}"

    for k in range(1, K):
        prob += (
                pulp.lpSum([Y[(k, i)] for i in range(1, n + 1)]) >=
                pulp.lpSum([Y[(k + 1, i)] for i in range(1, n + 1)])
        ), f"NoGap_{k}"

    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if i == j:
                continue
            for k in range(2, K + 1):
                prob += X[(i, j)] >= Y[(k - 1, i)] + Y[(k, j)] - 1, f"Link_{i}_{j}_k{k}"

    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if i == j:
                continue
            prob += Z[i] - Z[j] + (n + 1) * X[(i, j)] <= n, f"Subtour_{i}_{j}"

    for mv in must_visit_ids:
        if mv in pid_to_idx:
            i_idx = pid_to_idx[mv]
            if i_idx != 0:
                prob += pulp.lpSum([Y[(k, i_idx)] for k in range(1, K + 1)]) == 1, f"MustVisit_{mv}"

    distance_expr = pulp.lpSum([
        cost_mat[i][j] * X[(i, j)]
        for i in range(n + 1)
        for j in range(n + 1)
        if i != j
    ])
    prob += distance_expr <= daily_max_distance, "DailyMaxDistance"

    visit_time_terms = []
    for i in range(1, n + 1):
        pid = idx_to_pid[i]
        # Example: each place visited for 1 hour
        visit_duration = 1.0
        visit_time_terms.append(visit_duration * pulp.lpSum([Y[(k, i)] for k in range(1, K + 1)]))

    travel_time_terms = []
    for i in range(n + 1):
        for j in range(n + 1):
            if i == j:
                continue
            dist_ij = cost_mat[i][j]
            travel_time_terms.append((dist_ij / transport_speed_kmh) * X[(i, j)])

    total_time_expr = pulp.lpSum(visit_time_terms) + pulp.lpSum(travel_time_terms)
    prob += total_time_expr <= daily_max_duration, "DailyMaxDuration"
    prob += (
            pulp.lpSum([Y[(k, i)] for i in range(1, n + 1) for k in range(1, K + 1)])
            <= (max_places_per_day - 1)
    ), "MaxPlacesDay"

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if pulp.LpStatus[prob.status] not in ["Optimal", "Feasible"]:
        return [], 0.0, 0.0

    visited_sequence = []
    for k in range(1, K + 1):
        for i in range(1, n + 1):
            val = pulp.value(Y[(k, i)])
            if val and val > 0.5:
                visited_sequence.append(idx_to_pid[i])

    full_route = [start_pid] + visited_sequence + [start_pid]

    total_dist = 0.0
    for i in range(len(full_route) - 1):
        pidA = full_route[i]
        pidB = full_route[i + 1]
        idxA = pid_to_idx[pidA]
        idxB = pid_to_idx[pidB]
        dist_ab = cost_mat[idxA][idxB]
        total_dist += dist_ab

    total_travel_time = total_dist / transport_speed_kmh
    total_visit_time = len(visited_sequence) * 1.0
    total_dur = total_travel_time + total_visit_time

    return full_route, total_dist, total_dur


def calculate_itinerary(request_data: Dict[str, Any],
                        places_json_dir: str = "./app/data/") -> Tuple[List[Dict[str, Any]], float]:
    """
    Main function that takes a request_data dict, fetches the relevant tourist_spots from DB,
    then runs the existing cluster + MILP routine to produce an itinerary.
    """
    places_field = request_data.get("places")
    if places_field is None:
        raise ValueError("places 필드가 필요합니다.")
    if isinstance(places_field, int):
        db_nums = [places_field]
    elif isinstance(places_field, list):
        db_nums = places_field
    else:
        raise ValueError("places 필드는 정수 또는 정수 리스트여야 합니다.")

    # Instead of reading JSON files, fetch from DB:
    places_dict_total: Dict[str, Any] = {}
    for db_num in db_nums:
        partial_places = load_places_from_db(db_num)
        places_dict_total.update(partial_places)
    import logging
    logging.log(logging.DEBUG, f"{places_dict_total}")

    if not places_dict_total:
        import logging
        logging.log(logging.DEBUG, f"{places_dict_total}")
        raise ValueError("로드된 장소 데이터가 없습니다.")

    user_profile = request_data.get("user_profile", {})
    themes = user_profile.get("themes", [])
    must_visit_input = [str(x) for x in user_profile.get("must_visit_list", [])]
    not_visit_input = [str(x) for x in user_profile.get("not_visit_list", [])]
    if set(must_visit_input).intersection(not_visit_input):
        raise ValueError("must_visit과 not_visit 리스트에 중복된 항목이 있습니다.")

    valid_pids_all = list(places_dict_total.keys())
    valid_pids = [pid for pid in valid_pids_all if pid not in not_visit_input]

    num_days = request_data.get("num_days")
    max_places_per_day = request_data.get("max_places_per_day")
    daily_start_points_input = request_data.get("daily_start_points", [])
    if num_days is None or max_places_per_day is None or not isinstance(daily_start_points_input, list):
        raise ValueError("num_days, max_places_per_day, daily_start_points가 필요합니다.")
    if len(daily_start_points_input) < num_days:
        daily_start_points_input += [None] * (num_days - len(daily_start_points_input))
    else:
        daily_start_points_input = daily_start_points_input[:num_days]

    daily_max_distance = request_data.get("daily_max_distance", 99999)
    daily_max_duration = request_data.get("daily_max_duration", 99999)

    preferred_transport = user_profile.get("preferred_transport", "car")
    if preferred_transport == "walk":
        speed_kmh = 5.0
    elif preferred_transport == "public_transport":
        speed_kmh = 20.0
    else:
        speed_kmh = 60.0

    # Assign prizes
    base_prz, prio_prz = assign_prizes(
        places_dict_total, valid_pids,
        base_scale=0.1,
        priority_scale=2.0,
        cat_keywords=themes,
        cat_bonus=100.0
    )

    coords = [(places_dict_total[pid]["x"], places_dict_total[pid]["y"]) for pid in valid_pids]
    if not coords:
        return [], 0.0

    random.seed(42)
    chosen_indices = random.sample(range(len(coords)), num_days)
    centers = [coords[i] for i in chosen_indices]
    kmeans = KMeans(n_clusters=num_days, init=centers, n_init=1, random_state=42)
    labels = kmeans.fit_predict(coords)

    clusters: Dict[int, List[str]] = {i: [] for i in range(num_days)}
    for pid, ci in zip(valid_pids, labels):
        clusters[ci].append(pid)

    cluster_sequence = list(range(num_days))
    daily_itineraries: List[Dict[str, Any]] = []
    overall_distance = 0.0

    for day_idx, cluster_idx in enumerate(cluster_sequence):
        cluster_places = clusters.get(cluster_idx, [])
        if not cluster_places:
            continue

        stated_start_pid = daily_start_points_input[day_idx]
        if stated_start_pid and str(stated_start_pid) in places_dict_total:
            start_place = str(stated_start_pid)
            if start_place not in cluster_places:
                cluster_places.append(start_place)
        else:
            # Pick the place with the highest base_prz as the start
            start_place = max(cluster_places, key=lambda pid: base_prz.get(pid, 0.0))

        cluster_must_visits = [pid for pid in must_visit_input if pid in cluster_places and pid != start_place]

        route, day_dist, day_dur = solve_day_ptppp_milp(
            places_dict=places_dict_total,
            day_places=cluster_places,
            start_pid=start_place,
            daily_max_distance=daily_max_distance,
            daily_max_duration=daily_max_duration,
            max_places_per_day=max_places_per_day,
            base_prz=base_prz,
            prio_prz=prio_prz,
            must_visit_ids=cluster_must_visits,
            transport_speed_kmh=speed_kmh
        )

        if not route:
            daily_itineraries.append({
                "day": day_idx + 1,
                "route": [],
                "daily_distance": 0.0,
                "daily_duration": 0.0
            })
            continue

        def safe_int(x_str):
            try:
                return int(x_str)
            except:
                return x_str

        final_route_ids = [safe_int(x) for x in route]
        daily_itineraries.append({
            "day": day_idx + 1,
            "route": final_route_ids,
            "daily_distance": round(day_dist, 2),
            "daily_duration": round(day_dur, 2),
        })
        overall_distance += day_dist

    overall_distance = round(overall_distance, 2)
    return daily_itineraries, overall_distance
