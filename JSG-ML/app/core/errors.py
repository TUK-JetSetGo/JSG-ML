from typing import Awaitable

from fastapi import Request
from starlette.responses import Response, JSONResponse


class APIException(Exception):
    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message


async def exception_handler(request: Request, exc: Exception) -> Response:
    if isinstance(exc, APIException):
        return JSONResponse(
            status_code=exc.code,
            content={"code": exc.code, "message": exc.message, "data": None}
        )

    return JSONResponse(
        status_code=500,
        content={"code": 500, "message": "Internal Server Error", "data": None}
    )