import logging
from fastapi import FastAPI
from app.api.v1.endpoints import itinary
from app.core.errors import exception_handler, APIException

# 로거 설정
logging.basicConfig(
    level=logging.DEBUG,  # 디버깅 메시지도 출력하도록
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="JSG-ML",
    description="JSG-ML 연산서버 cicd test",
    version="1.0.0",
)
app.include_router(itinary.router, prefix="/api/v1")
app.add_exception_handler(APIException, exception_handler)

logger.info("FastAPI 앱이 시작되었습니다.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", reload=True)