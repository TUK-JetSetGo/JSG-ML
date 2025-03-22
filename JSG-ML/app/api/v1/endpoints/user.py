from fastapi import APIRouter

from app.core.exception.user_exception import UserNotFoundException
from app.schmes.base import BaseResponse
from app.schmes.user import UserResponse
from app.services.user_service import get_user_by_id

router = APIRouter()


@router.get("/users/{user_id}", response_model=BaseResponse[UserResponse])
async def read_user(user_id: int):
    user = get_user_by_id(user_id)
    if not user:
        raise UserNotFoundException(user_id)
    return BaseResponse(data=user)
