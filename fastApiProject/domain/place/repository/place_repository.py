import os
import json
from typing import List, Dict, Any
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[3]
DEFAULT_DATA_PATH = os.path.join(PROJECT_ROOT, "data")


class PlaceRepository:
    """
    data/{some_id}_관광지.json 파일을 읽어
    관광지 정보(리스트)를 반환하는 임시 레포지토리.
    """

    def __init__(self, base_path: str = None):
        """
        base_path를 지정하지 않으면,
        현재 모듈 파일 기준 상위 폴더(프로젝트 루트)에 있는 data 폴더를 사용.
        """
        if not base_path:
            base_path = DEFAULT_DATA_PATH
        self.base_path = os.path.abspath(base_path)

    def get_place_list_by_id(self, location_id: str) -> List[Dict[str, Any]]:
        """
        'data/{location_id}_관광지.json' 에서
        관광지 정보를 읽어 (list[dict])로 반환.
        """
        file_name = f"{location_id}_관광지.json"
        file_path = os.path.join(self.base_path, file_name)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # data 자체가 이미 관광지 정보 리스트라 가정
        return data

    def get_place_by_id(self, location_id: str, place_id: str):
        """
        'data/{place_id}_관광지.json' 에서
        관광지 정보를 읽어 (list[dict])로 반환.
        """
        places = self.get_place_list_by_id(location_id)
        for place in places:
            if place.get('id') == place_id:
                return place

        return None


# 사용 예시
if __name__ == "__main__":
    # base_path 인자를 지정하지 않으면 기본값(DEFAULT_DATA_PATH) 사용
    repo = PlaceRepository()
    place_id = "1"
    places_data = repo.get_place_list_by_id(place_id)
    print(places_data)
