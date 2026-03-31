from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

from server.auth import _make_token, verify_password
from server.config import ACCESS_PASSWORD

router = APIRouter()


class LoginRequest(BaseModel):
    password: str

    @field_validator("password")
    @classmethod
    def password_max_length(cls, v: str) -> str:
        if len(v) > 256:
            raise ValueError("password too long")
        return v


@router.post("/login")
async def login(req: LoginRequest):
    # 常量时间密码比较，防止时序攻击
    if verify_password(req.password):
        token = _make_token(ACCESS_PASSWORD)
        resp = JSONResponse({"ok": True})
        resp.set_cookie(
            "kb_auth", token,
            max_age=30 * 86400,
            httponly=True,
            samesite="lax",
            path="/",
        )
        return resp
    return JSONResponse({"ok": False}, status_code=403)
