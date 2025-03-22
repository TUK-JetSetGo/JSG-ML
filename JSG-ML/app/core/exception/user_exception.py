from app.core.errors import APIException


class UserNotFoundException(APIException):
    def __init__(self, user_id: int):
        super().__init__(code=404, message=f"{user_id} 는 존재하지 않는 사용자입니다.")
