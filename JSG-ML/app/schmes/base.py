from pydantic.generics import GenericModel
from typing import Generic, TypeVar, Optional

DataT = TypeVar("DataT")


class BaseResponse(GenericModel, Generic[DataT]):
    code: int = 200
    message: str = "success"
    data: Optional[DataT] = None
