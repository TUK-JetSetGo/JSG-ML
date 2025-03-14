# domain/itinarity/dto/itinarity_response_dto.py
from pydantic import BaseModel, Field
from typing import List


class MipOptimizationResponseDTO(BaseModel):
    """
    MIP 기반 최적화 응답 DTO
    - status: 최적화 결과 상태 (예: "Optimal")
    - objective: 목적 함수 최종 값 (만족도 점수 - 이동 비용)
    - route_indices: (순서, 관광지 인덱스) 쌍 리스트
    - route_place_ids: 최종 선택된 관광지 ID 순서
    """
    status: str = Field(..., example="Optimal")
    objective: float = Field(..., example=250.0)
    route_indices: List[List[int]] = Field(..., example=[[0, 1], [1, 0]])
    route_place_ids: List[int] = Field(..., example=[101, 102])


class TabuSearchResponseDTO(BaseModel):
    """
    Tabu Search 기반 최적화 응답 DTO
    - best_route_places: 최적 경로에 포함된 관광지 이름 리스트
    - objective: 최적 경로의 목표 함수 값
    """
    best_route_place_ids: List[int] = Field(..., example=[101, 102])
    objective: float = Field(..., example=180.0)
