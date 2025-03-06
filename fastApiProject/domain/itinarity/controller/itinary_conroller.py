# domain/itinarity/controller/itinary_conroller.py
from fastapi import APIRouter, HTTPException
from domain.itinarity.dto.itinarity_request_dto import MipOptimizationRequestDTO, TabuSearchRequestDTO
from domain.itinarity.dto.itinarity_response_dto import MipOptimizationResponseDTO, TabuSearchResponseDTO
from domain.itinarity.service.itinarity_service import solve_itinerary_mip, tabu_search_itinerary
from globals.config.response_config import APIResponse, create_success_response

router = APIRouter()

@router.post("/mip", response_model=APIResponse[MipOptimizationResponseDTO])
async def optimize_itinerary_mip(request: MipOptimizationRequestDTO):
    try:
        # 각 PlaceDTO 객체를 dict로 변환 후 전달
        places = [p.dict() for p in request.places]
        result = solve_itinerary_mip(places, request.user_profile.dict())
        return create_success_response(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tabu", response_model=APIResponse[TabuSearchResponseDTO])
async def optimize_itinerary_tabu(request: TabuSearchRequestDTO):
    try:
        places = [p.dict() for p in request.places]
        result = tabu_search_itinerary(places, request.user_profile.dict(), request.max_iter)
        return create_success_response(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))