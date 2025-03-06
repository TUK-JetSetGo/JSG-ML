# app/exceptions/global_exception.py
from fastapi import Request
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger("global_exception")


# 커스텀 예외 클래스
class GlobalException(Exception):
    def __init__(self, detail: str):
        self.detail = detail


# 글로벌 예외 핸들러 함수
async def global_exception_handler(request: Request, exc: GlobalException):
    logger.error(f"Unhandled GlobalException: {exc.detail}")
    return JSONResponse(
        status_code=500,
        content={
            "code": 500,
            "message": "Internal Server Error",
            "detail": exc.detail,
        },
    )
