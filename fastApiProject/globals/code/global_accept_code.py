from enum import Enum


class GlobalAcceptCode(Enum):
    SUCCESS = 200
    VALIDATION_ERROR = 401
    NOT_FOUND = 404
    SERVER_ERROR = 500

    def message(self):
        messages = {
            GlobalAcceptCode.SUCCESS: "성공입니다",
            GlobalAcceptCode.VALIDATION_ERROR: "금지된 요청입니다",
            GlobalAcceptCode.NOT_FOUND: "리소스를 찾을 수 없습니다",
            GlobalAcceptCode.SERVER_ERROR: "서버오류",
        }
        return messages.get(self, "알 수 없는 오류가 발생했습니다")
