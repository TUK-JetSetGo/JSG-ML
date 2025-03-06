# main.py
from fastapi import FastAPI
from domain.itinarity.controller.itinary_conroller import router as itinarity_router
from globals.config.swagger_config import custom_openapi
from globals.exception.global_exception import GlobalException, global_exception_handler

app = FastAPI(title="JSG_ML")

# 글로벌 예외 핸들러 등록
app.add_exception_handler(GlobalException, global_exception_handler)

# 도메인 컨트롤러(라우터) 포함
app.include_router(itinarity_router, prefix="/itinarity", tags=["Itinarity"])

# Swagger 커스터마이징 (custom OpenAPI 스키마 적용)
app.openapi = lambda: custom_openapi(app)

app.include_router(itinarity_router, prefix="/itinarity", tags=["Itinarity"])

# 기본 엔드포인트 추가 (옵션)
@app.get("/")
async def root():
    return {"message": "Welcome to the FastAPI Project"}

if __name__ == "__main__":
    import uvicorn
    # "main:app" 방식으로 모듈과 변수 이름을 지정하여 실행
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)