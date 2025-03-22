from fastapi import APIRouter, HTTPException

from app.schmes.base import BaseResponse
from app.schmes.itinary import ItinaryResponse, ItinaryRequest
from app.services.itinary_service import calculate_itinerary

router = APIRouter(prefix="/itinary")


@router.post("/", response_model=BaseResponse[ItinaryResponse])
async def calculate_best_route(request: ItinaryRequest):
    try:
        daily_itineraries, overall_distance = calculate_itinerary(request.dict())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    response_data = {
        "daily_itineraries": daily_itineraries,
        "overall_distance": overall_distance
    }
    return BaseResponse(data=response_data)
