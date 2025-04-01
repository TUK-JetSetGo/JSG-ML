import os
import json
import folium
import pymysql
from typing import Dict, Any
from dotenv import load_dotenv

# .env 파일에 있는 환경변수를 로드합니다.
load_dotenv()

# 데이터베이스 접속 정보
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


def get_initial_coordinates(route, places_dict):
    """
    route 리스트에서 첫 번째 유효한 장소 좌표를 찾습니다.
    """
    for pid in route:
        pid_str = str(pid)
        if pid_str in places_dict:
            # DB에서는 x, y를 사용했으므로 y가 위도, x가 경도
            return places_dict[pid_str]["y"], places_dict[pid_str]["x"]
    return None, None


def plot_itinerary_on_map(places_dict, daily_itineraries, output_html="itinerary_map.html"):
    """
    daily_itineraries 예시:
    [
      {
        "day": 1,
        "route": [36183644, 1985618339, 38484323, 20947594, 36183644],
        "daily_distance": 46.8,
        "daily_duration": 3.78
      },
      ...
    ]

    각 일자의 경로를 서로 다른 색상으로 지도에 표시하며,
    각 정점에 마커를 추가하여 장소 이름, 카테고리 정보와 함께
    일자의 총 이동거리 및 소요시간을 팝업으로 보여줍니다.
    """
    if not daily_itineraries:
        print("No itinerary data provided.")
        return

    # 첫 날의 유효한 좌표 찾기
    first_day = daily_itineraries[0]
    if not first_day.get("route"):
        print("첫째날의 경로가 비어있습니다.")
        return

    start_lat, start_lng = get_initial_coordinates(first_day["route"], places_dict)
    if start_lat is None:
        print("첫번째 유효한 장소 정보를 찾을 수 없습니다.")
        return

    m = folium.Map(location=[start_lat, start_lng], zoom_start=12)

    # 색상 리스트 (필요시 확장)
    colors = ["red", "blue", "green", "purple", "orange", "darkred",
              "lightred", "beige", "darkblue", "darkgreen", "cadetblue",
              "darkpurple", "white", "pink", "lightblue", "lightgreen",
              "gray", "black", "lightgray"]

    for itinerary in daily_itineraries:
        day = itinerary.get("day")
        route = itinerary.get("route", [])
        daily_distance = itinerary.get("daily_distance", "N/A")
        daily_duration = itinerary.get("daily_duration", "N/A")
        if not route:
            continue

        coords = []
        for pid in route:
            pid_str = str(pid)
            if pid_str not in places_dict:
                print(f"Warning: Place ID {pid_str} not found. Skipping.")
                continue
            # DB에서는 y, x를 사용 (y: 위도, x: 경도)
            lat = places_dict[pid_str]["y"]
            lng = places_dict[pid_str]["x"]
            coords.append((lat, lng))
        if not coords:
            continue

        color = colors[(day - 1) % len(colors)]
        polyline_tooltip = f"Day {day} Route: {daily_distance} km, {daily_duration} h"
        folium.PolyLine(coords, tooltip=polyline_tooltip, color=color, weight=5).add_to(m)

        # 각 경유지에 마커 추가
        for i, (lat, lng) in enumerate(coords):
            pid = route[i]
            info = places_dict.get(str(pid), {})
            name = info.get("name", "Unknown")
            category = info.get("category", [])
            cat_str = ", ".join(category) if category else ""
            if i == 0:
                popup_text = f"Day {day} Start: {name} ({cat_str})"
            elif i == len(coords) - 1:
                popup_text = f"Day {day} End: {name} ({cat_str})"
            else:
                popup_text = f"Day {day} Stop {i}: {name} ({cat_str})"
            folium.Marker(
                location=[lat, lng],
                popup=popup_text,
                tooltip=popup_text,
                icon=folium.Icon(color=color)
            ).add_to(m)

    m.save(output_html)
    print(f"Map has been saved to {output_html}")


if __name__ == "__main__":
    # 예시 응답 JSON (daily_itineraries 데이터)
    sample_response = {
        "code": 200,
        "message": "success",
        "data": {
            "daily_itineraries": [
                {
                    "day": 1,
                    "route": [
                        36183644,
                        1985618339,
                        38484323,
                        20947594,
                        36183644
                    ],
                    "daily_distance": 46.8,
                    "daily_duration": 3.78
                },
                {
                    "day": 2,
                    "route": [
                        1592500971,
                        13491575,
                        13491073,
                        11491281,
                        1592500971
                    ],
                    "daily_distance": 47.2,
                    "daily_duration": 3.79
                },
                {
                    "day": 3,
                    "route": [
                        10983870,
                        13350575,
                        1130170517,
                        1892836085,
                        10983870
                    ],
                    "daily_distance": 57.17,
                    "daily_duration": 3.95
                }
            ],
            "overall_distance": 151.18
        }
    }

    # city_id를 실제 값에 맞게 지정합니다.
    city_id = 9
    places_dict = load_places_from_db(city_id)
    daily_itineraries = sample_response["data"]["daily_itineraries"]
    plot_itinerary_on_map(places_dict, daily_itineraries, output_html="itinerary_map.html")
