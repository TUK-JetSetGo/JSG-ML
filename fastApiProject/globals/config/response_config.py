# globals/config/response_config.py
from pydantic import BaseModel
from typing import Generic, TypeVar, Optional
from pydantic.generics import GenericModel

from globals.code.global_accept_code import GlobalAcceptCode

DataT = TypeVar("DataT")

class APIResponse(GenericModel, Generic[DataT]):
    code: int
    message: str
    data: Optional[DataT] = None

def create_success_response(data: DataT) -> APIResponse:
    return APIResponse(
        code=GlobalAcceptCode.SUCCESS.value,
        message=GlobalAcceptCode.SUCCESS.message(),
        data=data
    )