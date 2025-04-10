from pydantic import BaseModel
from typing import List, Optional


class UserProfile(BaseModel):
    travel_type: str
    group_size: int
    budget_amount: int
    themes: List[str]
    must_visit_list: List[int]
    not_visit_list: List[int]
    preferred_transport: str


class ItinaryRequest(BaseModel):
    places: List[int]
    user_profile: UserProfile
    max_iter: int
    num_days: int
    max_places_per_day: int
    daily_start_points: List[Optional[int]]
    daily_max_distance: Optional[float] = None
    daily_max_duration: Optional[float] = None


class DayItinerary(BaseModel):
    day: int
    route: List[int]
    daily_distance: float
    daily_duration: float


class ItinaryResponse(BaseModel):
    daily_itineraries: List[DayItinerary]
    overall_distance: float
