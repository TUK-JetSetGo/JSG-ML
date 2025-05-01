import logging
import math
import random
import json
from collections import defaultdict
from functools import lru_cache
from typing import List, Dict, Any, Tuple

import requests
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
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

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # 또는 DEBUG


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


# 한 번만 생성해서 커넥션을 재사용
_session = requests.Session()


@lru_cache(maxsize=None)
def compute_distance_cached(lon1, lat1, lon2, lat2):
    # OSRM 직접 호출
    try:
        url = f"http://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?overview=false"
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        return data["routes"][0]["distance"] / 1000.0
    except Exception:
        # 해버사인 폴백
        rad = math.pi / 180
        dlat = (lat2 - lat1) * rad
        dlon = (lon2 - lon1) * rad
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1 * rad) * math.cos(lat2 * rad) * math.sin(dlon / 2) ** 2
        return 6371 * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """
    두 경위도(lon1,lat1) ↔ (lon2,lat2) 사이의 대원거리(km)를 반환
    """
    R = 6371.0  # 지구 반경(km)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def compute_distance(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    """
    원래 형태 그대로 쓰되, 내부에서 좌표만 꺼내 캐시된 함수 호출.
    """
    lon1, lat1 = a["x"], a["y"]
    lon2, lat2 = b["x"], b["y"]
    return haversine(lon1, lat1, lon2, lat2)


def build_cost_matrix(places_dict: Dict[str, Any], place_ids: List[str]) -> List[List[float]]:
    """
    place_ids 순서대로 해버사인 거리(km)로만 N×N 매트릭스를 채워 반환.
    """
    # 좌표 리스트
    coords = [(places_dict[pid]["x"], places_dict[pid]["y"]) for pid in place_ids]
    n = len(coords)
    M = [[0.0] * n for _ in range(n)]
    for i in range(n):
        lon1, lat1 = coords[i]
        for j in range(n):
            if i == j:
                continue
            lon2, lat2 = coords[j]
            M[i][j] = haversine(lon1, lat1, lon2, lat2)
    return M


def build_cost_matrix_with_table(places_dict: Dict[str, Any],
                                 place_ids: List[str]) -> List[List[float]]:
    """
    OSRM Table API로 한 번에 N×N 거리(km) 행렬을 받아옵니다.
    """
    coords = ";".join(
        f"{places_dict[pid]['x']},{places_dict[pid]['y']}"
        for pid in place_ids
    )
    url = f"http://router.project-osrm.org/table/v1/driving/{coords}?annotations=distance"
    resp = _session.get(url, timeout=10)
    resp.raise_for_status()
    dist_m = resp.json()["distances"]  # meters
    n = len(dist_m)
    return [[dist_m[i][j] / 1000.0 for j in range(n)] for i in range(n)]


def assign_prizes(
        places_dict: Dict[str, Any],
        place_ids: List[str],
        must_visit_list: List[str] = None,
        base_scale: float = 1.0,
        priority_scale: float = 0.3,
        cat_keywords: List[str] = None,
        cat_bonus: float = 10.0,
        must_bonus: float = 1e6,
) -> Tuple[Dict[str, float], Dict[int, Dict[str, float]]]:
    """
    - must_visit_list: 반드시 포함해야 할 장소 ID 리스트
    - must_bonus: must_visit 은 base_val에 이만큼 더해져 항상 최상위로 랭크됨
    """
    must_visit_list = must_visit_list or []

    base_prize: Dict[str, float] = {}
    priority_prize: Dict[int, Dict[str, float]] = {}

    for pid in place_ids:
        info = places_dict[pid]
        rc = info.get("reviewCount", 0)
        cat_match = 0
        if cat_keywords:
            for kw in cat_keywords:
                if kw in info.get("category", ""):
                    cat_match = 1
                    break

        # 기본 점수 산출
        base_val = rc * base_scale
        if cat_match:
            base_val += cat_bonus

        # must_visit 보너스
        if pid in must_visit_list:
            base_val += must_bonus

        base_prize[pid] = base_val

    # 순서 k 에 따른 우선순위 점수 계산
    for k in range(1, len(place_ids) + 1):
        priority_prize[k] = {}
        for pid in place_ids:
            p_base = base_prize.get(pid, 0.0)
            priority_prize[k][pid] = p_base * priority_scale / math.sqrt(k)

    return base_prize, priority_prize


D_THRESH = 30.0  # km


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
        cost_mat: List[List[float]],
        transport_speed_kmh: float = 40.0,
) -> Tuple[List[str], float, float]:
    if not day_places:
        return [], 0.0, 0.0

    # Remove duplicates
    day_places = list(set(day_places))
    if start_pid not in day_places:
        day_places.append(start_pid)

    for mv in must_visit_ids:
        if mv not in day_places:
            day_places.append(mv)

    filtered = []
    for pid in day_places:
        if pid == start_pid or pid in must_visit_ids:
            filtered.append(pid)
        else:
            dist = compute_distance(places_dict[start_pid], places_dict[pid])
            if dist <= D_THRESH:
                filtered.append(pid)
    day_places = filtered

    if len(day_places) > max_places_per_day:
        # Exclude start_pid, sort the rest by base_prz, and keep top
        musts = [pid for pid in day_places
                 if pid in must_visit_ids and pid != start_pid]
        others = [pid for pid in day_places
                  if pid not in musts and pid != start_pid]
        others.sort(key=lambda x: base_prz.get(x, 0.0), reverse=True)
        cap = max_places_per_day - 1 - len(musts)
        chosen = others[:cap]
        day_places = [start_pid] + musts + chosen

    place_ids = [start_pid] + [pid for pid in day_places if pid != start_pid]
    n = len(place_ids) - 1
    K = min(n, max_places_per_day - 1)

    idx_to_pid = {0: start_pid}
    for i, pid in enumerate(place_ids[1:], start=1):
        idx_to_pid[i] = pid
    pid_to_idx = {v: k for k, v in idx_to_pid.items()}
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

    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=30))

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
    logger.info(f"🔎 Day {start_pid} Route: {full_route}")
    logger.info(f"📏 place_ids: {place_ids}")
    logger.info(f"🧭 pid_to_idx: {pid_to_idx}")
    for i in range(len(full_route) - 1):
        pidA = full_route[i]
        pidB = full_route[i + 1]
        idxA = pid_to_idx[pidA]
        idxB = pid_to_idx[pidB]
        if idxA is None or idxB is None:
            logger.warning(f"⚠️ pid_to_idx missing: {pidA}→{idxA}, {pidB}→{idxB}")
            continue

        dist_ab = cost_mat[idxA][idxB]
        logger.info(f"↔️ {pidA} → {pidB} = {dist_ab:.2f} km (idx {idxA}->{idxB})")
        total_dist += dist_ab

    total_travel_time = total_dist / transport_speed_kmh
    total_visit_time = len(visited_sequence) * 1.0
    total_dur = total_travel_time + total_visit_time

    return full_route, total_dist, total_dur


def solve_one_day(args):
    # unpack args including all_place_ids, full_cost_mat
    (day_idx,
     cluster_places,
     start_pid,
     daily_max_distance,
     daily_max_duration,
     max_places_per_day,
     base_prz,
     prio_prz,
     must_visit_ids,
     speed_kmh,
     places_dict,
     all_place_ids,
     full_cost_mat) = args

    # ——————————————
    # 1) 이 날 방문할 place_ids 리스트 구성
    day_place_ids = [start_pid] + [pid for pid in cluster_places if pid != start_pid]

    # 2) all_place_ids 에서 이 날 장소들의 인덱스만 뽑아내서
    idxs = [all_place_ids.index(pid) for pid in day_place_ids]

    # 3) full_cost_mat 에서 부분 행렬(slice) 생성
    cost_mat = [
        [full_cost_mat[i][j] for j in idxs]
        for i in idxs
    ]
    # ——————————————

    # 4) MILP 실행할 때 cost_mat 을 넘겨줌
    route, day_dist, day_dur = solve_day_ptppp_milp(
        places_dict=places_dict,
        day_places=cluster_places,
        start_pid=start_pid,
        daily_max_distance=daily_max_distance,
        daily_max_duration=daily_max_duration,
        max_places_per_day=max_places_per_day,
        base_prz=base_prz,
        prio_prz=prio_prz,
        must_visit_ids=must_visit_ids,
        transport_speed_kmh=speed_kmh,
        cost_mat=cost_mat  # ← 여기에 추가
    )

    # 5) 로그 출력
    print(f"🔹 Day {day_idx + 1} generated → "
          f"route={route}, dist={day_dist:.2f}km, dur={day_dur:.2f}h")

    # 6) 결과 반환
    return day_idx, (route, day_dist, day_dur)


def calculate_itinerary(request_data: Dict[str, Any],
                        places_json_dir: str = "./app/data/") -> Tuple[List[Dict[str, Any]], float]:
    """
    Main function that takes a request_data dict, fetches the relevant tourist_spots from DB,
    then runs the existing cluster + MILP routine to produce an itinerary.
    """
    import time
    t0 = time.perf_counter()

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

    t1 = time.perf_counter()
    logger.info(f"[Timing] load_places_from_db 총 {(t1 - t0):.2f}s")

    if not places_dict_total:
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

    daily_max_distance = 200  # request_data.get("daily_max_distance", 99999)
    daily_max_duration = 20  # request_data.get("daily_max_duration", 99999)

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
    t2 = time.perf_counter()
    logger.info(f"[Timing] assign_prizes {(t2 - t1):.2f}s")

    coords = [(places_dict_total[pid]["x"], places_dict_total[pid]["y"])
              for pid in valid_pids]
    if not coords:
        return [], 0.0

    seed_coords: List[Tuple[float, float]] = []
    for pid in must_visit_input:
        if pid in valid_pids:
            seed_coords.append((
                places_dict_total[pid]["x"],
                places_dict_total[pid]["y"]
            ))

    # 2) 모자라면 나머지 coords 에서 랜덤 채우기
    if len(seed_coords) < num_days:
        others = [c for c in coords if c not in seed_coords]
        random.seed(42)
        seed_coords += random.sample(others, num_days - len(seed_coords))
    else:
        seed_coords = seed_coords[:num_days]

    # 3) 씨딩된 센터로 KMeans 실행 (n_init=1)
    kmeans = KMeans(
        n_clusters=num_days,
        init=seed_coords,
        n_init=1,
        random_state=42
    )
    labels = kmeans.fit_predict(coords)
    place_to_cluster = {pid: ci for pid, ci in zip(valid_pids, labels)}

    cluster_to_must = defaultdict(list)
    for pid in must_visit_input:
        ci = place_to_cluster.get(pid)
        if ci is not None:
            cluster_to_must[ci].append(pid)

    day_to_must = {i: [] for i in range(num_days)}
    for i, (cluster_id, musts) in enumerate(cluster_to_must.items()):
        for j, pid in enumerate(musts):
            day = (i + j) % num_days
            day_to_must[day].append(pid)

    t3 = time.perf_counter()
    logger.info(f"[Timing] KMeans.fit_predict {(t3 - t2):.2f}s")

    # ───────────────────────────────────────────────────────

    clusters: Dict[int, List[str]] = {i: [] for i in range(num_days)}
    for pid, ci in zip(valid_pids, labels):
        clusters[ci].append(pid)

    for cid, pids in clusters.items():
        hits = set(pids) & set(must_visit_input)
        logger.info(f"[Cluster {cid}] total={len(pids)} places, must_visit_hits={hits}")

    all_place_ids = valid_pids[:]
    t4 = time.perf_counter()
    full_cost_mat = build_cost_matrix(places_dict_total, all_place_ids)
    t5 = time.perf_counter()
    logger.info(f"[Timing] build_cost_matrix_with_table {(t5 - t4):.2f}s")

    daily_args = []
    t_days_start = time.perf_counter()
    for day_idx in range(num_days):
        cluster_places = clusters.get(day_idx, [])
        provided = daily_start_points_input[day_idx]
        if provided:
            # 유저가 직접 지정한 출발점이 있으면 그대로 사용
            start_pid = provided
        else:
            # cluster_places 중 category가 "숙박" 또는 "호텔"인 곳만 추려내기
            lodging = [
                pid for pid in cluster_places
                if any(
                    (isinstance(cat, str) and cat in ["숙박", "호텔"]) or
                    (isinstance(cat, dict) and cat.get("name") in ["숙박", "호텔"])
                    for cat in places_dict_total[pid]["category"]
                )
            ]
            if lodging:
                # 후보가 있으면 base_prz(우선순위 점수) 최고 지점을 start로
                start_pid = max(lodging, key=lambda pid: base_prz.get(pid, 0.0))
            else:
                # 없으면 기존대로 전체 중 base_prz 최고 지점을 start로
                start_pid = max(cluster_places, key=lambda pid: base_prz.get(pid, 0.0))

        must_visits = day_to_must[day_idx]
        daily_args.append((
            day_idx,
            cluster_places,
            start_pid,
            daily_max_distance,
            daily_max_duration,
            max_places_per_day,
            base_prz,
            prio_prz,
            must_visits,
            speed_kmh,
            places_dict_total,
            all_place_ids,
            full_cost_mat
        ))

    t_days_end = time.perf_counter()
    logger.info(f"[Timing] solve_one_day 전체 {(t_days_end - t_days_start):.2f}s")

    daily_itineraries: List[Dict[str, Any]] = []
    overall_distance = 0.0
    with ProcessPoolExecutor(max_workers=15) as pool:
        futures = {pool.submit(solve_one_day, args): args[0] for args in daily_args}
        for fut in as_completed(futures):
            day_idx = futures[fut]
            route, day_dist, day_dur = fut.result()[1]

            if not route or len(route) <= 2:
                logger.warning(f"⚠️ Day {day_idx + 1} route empty. Applying fallback...")

                (
                    _,
                    cluster_places,
                    start_pid,
                    daily_max_distance,
                    daily_max_duration,
                    max_places_per_day,
                    base_prz,
                    prio_prz,
                    must_visits,
                    speed_kmh,
                    places_dict_total,
                    all_place_ids,
                    _
                ) = daily_args[day_idx]

                fallback_pids = []
                visited_must_visits = set()
                for itin in daily_itineraries:
                    visited_must_visits.update(itin["route"])

                # 아직 방문하지 않은 must_visit 우선
                unvisited_must_visits = [
                    pid for pid in must_visits if pid != start_pid and pid not in visited_must_visits
                ]

                for pid in cluster_places:
                    if pid == start_pid  or pid in must_visits:
                        continue
                    dist = compute_distance(places_dict_total[start_pid], places_dict_total[pid])
                    if dist <= daily_max_distance:
                        score = base_prz.get(pid, 0.0)
                        fallback_pids.append((pid, score))

                # must_visit 먼저 선택
                must_visit_selected = [pid for pid in unvisited_must_visits if any(pid == p for p, _ in fallback_pids)]
                remaining_capacity = max_places_per_day - 1 - len(must_visit_selected)

                # 나머지 base_prz 높은 순으로 채움
                non_must = [(pid, score) for pid, score in fallback_pids if pid not in must_visit_selected]
                non_must.sort(key=lambda x: x[1], reverse=True)
                non_must_selected = [pid for pid, _ in non_must[:remaining_capacity]]

                selected = must_visit_selected + non_must_selected
                fallback_route = [start_pid] + selected + [start_pid]

                # 거리 및 시간 계산
                fallback_dist = 0.0
                for i in range(len(fallback_route) - 1):
                    pidA = fallback_route[i]
                    pidB = fallback_route[i + 1]
                    fallback_dist += compute_distance(places_dict_total[pidA], places_dict_total[pidB])

                fallback_dur = fallback_dist / speed_kmh + len(selected) * 1.0  # 관광지별 1시간

                daily_itineraries.append({
                    "day": day_idx + 1,
                    "route": [int(x) for x in fallback_route],
                    "daily_distance": round(fallback_dist, 2),
                    "daily_duration": round(fallback_dur, 2),
                })
                overall_distance += fallback_dist
                continue


            daily_itineraries.append({
                "day": day_idx + 1,
                "route": [int(x) for x in route],
                "daily_distance": round(day_dist, 2),
                "daily_duration": round(day_dur, 2),
            })
            overall_distance += day_dist

    visited = set()
    for itin in daily_itineraries:
        visited.update(itin["route"])

    unvisited_musts = [pid for pid in must_visits if pid not in visited]
    if unvisited_musts:
        logger.info(f"🔧 강제 삽입할 must_visit 남음: {unvisited_musts}")

        must_visit_set = set(must_visits)  # 상단에서 선언해두면 속도 개선

        for must_pid in unvisited_musts:
            inserted = False
            for itin in daily_itineraries:
                route = itin["route"]
                if must_pid in route:
                    inserted = True
                    break

                middle = route[1:-1]
                if len(middle) < max_places_per_day:
                    route.insert(-1, must_pid)
                    itin["route"] = route
                    inserted = True
                    logger.info(f"✅ must_visit {must_pid} 빈 칸에 삽입 완료 (day {itin['day']})")
                    break
                else:
                    # must_visit 제외한 곳 중 base_prz 가장 낮은 것 찾기
                    replace_candidates = [
                        pid for pid in middle if pid not in must_visit_set
                    ]
                    if not replace_candidates:
                        continue

                    min_prz_pid = min(
                        replace_candidates,
                        key=lambda pid: base_prz.get(pid, float("inf"))
                    )

                    idx = route.index(min_prz_pid)
                    route[idx] = must_pid
                    itin["route"] = route
                    inserted = True
                    logger.info(f"♻️ must_visit {must_pid}이 base_prz 낮은 관광지 {min_prz_pid}와 교체됨 (day {itin['day']})")
                    break

    overall_distance = round(overall_distance, 2)
    return daily_itineraries, overall_distance
