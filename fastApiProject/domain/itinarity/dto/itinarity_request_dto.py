# domain/itinarity/dto/itinarity_request_dto.py
from pydantic import BaseModel, Field
from typing import List, Optional

class UserProfileDTO(BaseModel):
    travel_type: str = Field(..., example="group")
    group_size: int = Field(..., example=4)
    budget_amount: float = Field(..., example=150000)
    themes: List[str] = Field(..., example=["미식", "여행,명소"])
    must_visit_list: List[int] = Field(..., example=[101])
    preferred_transport: str = Field(..., example="car")


class MipOptimizationRequestDTO(BaseModel):
    """
    MIP 기반 최적화 요청 DTO
    - places: 후보 관광지 리스트
    - user_profile: 사용자의 프로필 및 조건
    """
    places: List[int]
    user_profile: UserProfileDTO


class TabuSearchRequestDTO(BaseModel):
    """
    Tabu Search 기반 최적화 요청 DTO
    - places: 후보 관광지 리스트
    - user_profile: 사용자의 프로필 및 조건
    - max_iter: 최대 반복 횟수 (기본값 300)
    """
    places: List[int]
    user_profile: UserProfileDTO
    max_iter: Optional[int] = Field(300, example=300)
