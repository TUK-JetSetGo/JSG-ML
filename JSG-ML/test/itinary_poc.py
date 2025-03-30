import json
import folium


def load_places(json_path):
    """
    1_관광지.json을 불러와서,
    { place_id: {"id": ..., "name": ..., "lat": float, "lng": float, "category": [...]}, ... }
    형태로 리턴
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)  # data: List[Dict]
    places_dict = {}
    for item in data:
        pid = str(item["id"])  # 문자열 ID
        # x를 경도(lng), y를 위도(lat)로 가정 (필요에 따라 수정)
        lng = float(item["x"])
        lat = float(item["y"])
        name = item.get("name", f"Place_{pid}")
        # 카테고리(테마) 정보: 만약 문자열이면 리스트로 변환
        cat = item.get("category", [])
        if isinstance(cat, str):
            cat = [cat]
        places_dict[pid] = {
            "id": pid,
            "name": name,
            "lat": lat,
            "lng": lng,
            "category": cat
        }
    return places_dict


def plot_itinerary_on_map(places_dict, daily_itineraries, output_html="itinerary_map.html"):
    """
    daily_itineraries: [
        {
            "day": 1,
            "route": [19708047, 11576965, 11491638, 13491801, 19708047],
            "daily_distance": 1.4053806296885687
        },
        ...
    ]

    각 일자의 경로를 서로 다른 색상으로 지도에 표시하며,
    각 정점에 마커를 추가하여 장소 이름 및 카테고리 정보를 팝업으로 보여줍니다.
    """
    if not daily_itineraries:
        print("No itinerary data provided.")
        return

    # 초기 지도 위치는 첫 날의 시작 장소 좌표로 설정
    first_day = daily_itineraries[0]
    if not first_day.get("route"):
        print("첫째날의 경로가 비어있습니다.")
        return
    first_pid = str(first_day["route"][0])
    if first_pid not in places_dict:
        print("첫번째 장소 정보가 없습니다.")
        return
    start_lat = places_dict[first_pid]["lat"]
    start_lng = places_dict[first_pid]["lng"]
    m = folium.Map(location=[start_lat, start_lng], zoom_start=12)

    # 색상 리스트 (필요시 확장)
    colors = ["red", "blue", "green", "purple", "orange", "darkred",
              "lightred", "beige", "darkblue", "darkgreen", "cadetblue",
              "darkpurple", "white", "pink", "lightblue", "lightgreen",
              "gray", "black", "lightgray"]

    for itinerary in daily_itineraries:
        day = itinerary.get("day")
        route = itinerary.get("route", [])
        if not route:
            continue

        coords = []
        for pid in route:
            pid_str = str(pid)
            if pid_str not in places_dict:
                print(f"Warning: Place ID {pid_str} not found. Skipping.")
                continue
            lat = places_dict[pid_str]["lat"]
            lng = places_dict[pid_str]["lng"]
            coords.append((lat, lng))
        if not coords:
            continue

        color = colors[(day - 1) % len(colors)]
        folium.PolyLine(coords, tooltip=f"Day {day} Route", color=color, weight=5).add_to(m)

        # 마커 추가
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
    # 예시 응답 JSON (새로운 스키마에 따른 daily_itineraries)
    sample_response = {
  "code": 200,
  "message": "success",
  "data": {
    "daily_itineraries": [
      {
        "day": 1,
        "route": [
          11576965,
          13491073,
          12226320
        ],
        "daily_distance": 0.47,
        "daily_duration": 2.07
      },
      {
        "day": 2,
        "route": [
          12226320,
          19932820
        ],
        "daily_distance": 0.49,
        "daily_duration": 1.07
      },
      {
        "day": 3,
        "route": [
          19932820
        ],
        "daily_distance": 0,
        "daily_duration": 0
      },
      {
        "day": 4,
        "route": [
          19932820,
          11576965
        ],
        "daily_distance": 0.41,
        "daily_duration": 1.06
      }
    ],
    "overall_distance": 1.3769839092701246
  }
}

    # 예시: 실제 파일 경로에 맞게 수정
    json_path = "../app/data/1_관광지.json"
    places_dict = load_places(json_path)
    daily_itineraries = sample_response["data"]["daily_itineraries"]
    plot_itinerary_on_map(places_dict, daily_itineraries, output_html="itinerary_map.html")
