from fastapi import APIRouter, HTTPException, Depends
from typing import List
from domain.itinarity.dto.itinarity_request_dto import TabuSearchRequestDTO
from domain.itinarity.dto.itinarity_response_dto import TabuSearchResponseDTO
from domain.itinarity.service.itinarity_service import tabu_search_itinerary, optimize_tabu_search
from domain.place.repository.place_repository import PlaceRepository
from globals.config.response_config import APIResponse, create_success_response

router = APIRouter()


@router.post("/search", response_model=APIResponse[TabuSearchResponseDTO])
async def optimize_itinerary_tabu(request: TabuSearchRequestDTO):
    """
    Tabu Search 기반으로 일정 최적화 결과를 반환하는 엔드포인트.
    request DTO:
      - places: 관광지 ID 리스트
      - must_visit_places: 필수 방문지 ID 리스트
      - user_profile: 사용자의 프로필
      - max_iter: 반복 횟수
    """
    try:
        # Service 계층 호출
        result = optimize_tabu_search(request)
        return create_success_response(result)

    except Exception as e:
        # 예외 처리
        raise HTTPException(status_code=500, detail=str(e))
