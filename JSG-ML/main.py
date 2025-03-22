from fastapi import FastAPI
from app.api.v1.endpoints import itinary
from app.core.errors import exception_handler, APIException

app = FastAPI(
    title="JSG-ML",
    description="JSG-ML 연산 서버",
    version="1.0.0",
)
app.include_router(itinary.router, prefix="/api/v1")
app.add_exception_handler(APIException, exception_handler)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", reload=True)
