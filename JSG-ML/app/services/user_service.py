def get_user_by_id(user_id: int):
    if user_id != 1:
        return None

    return dict(id=1, name="홍길동", email="hong@gmail.com")
