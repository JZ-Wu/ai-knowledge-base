from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from server.auth import _make_token
from server.config import ACCESS_PASSWORD

router = APIRouter()


class LoginRequest(BaseModel):
    password: str


@router.post("/login")
async def login(req: LoginRequest):
    if req.password == ACCESS_PASSWORD:
        token = _make_token(ACCESS_PASSWORD)
        resp = JSONResponse({"ok": True})
        # cookie 有效期 30 天
        resp.set_cookie("kb_auth", token, max_age=30 * 86400, httponly=True, samesite="lax")
        return resp
    return JSONResponse({"ok": False}, status_code=403)
