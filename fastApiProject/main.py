# main.py
from fastapi import FastAPI
from domain.itinarity.controller.itinary_conroller import router as itinarity_router
from globals.config.swagger_config import custom_openapi
from globals.exception.global_exception import GlobalException, global_exception_handler

app = FastAPI(title="JSG_ML")
app.add_exception_handler(GlobalException, global_exception_handler)
app.include_router(itinarity_router, prefix="/itinarity", tags=["Itinarity"])
app.openapi = lambda: custom_openapi(app)

@app.get("/")
async def root():
    return {"message": "Welcome to the FastAPI Project"}

app.include_router(itinarity_router, prefix="/itinarity", tags=["Itinarity"])


if __name__ == "__main__":
    import uvicorn
    # "main:app" 방식으로 모듈과 변수 이름을 지정하여 실행
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)