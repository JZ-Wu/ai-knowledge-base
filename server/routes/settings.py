"""GET / PUT 运行时设置 + 登录接口（之前在 routes/auth.py，归并过来）。

GET /api/settings → 脱敏后的设置（API key 只返回 configured 标记）
PUT /api/settings → 完整覆盖写入（接收完整结构，含 API key）
GET /api/settings/check → 仅返回 is_configured + claude_cli_available，用于首次启动决策
POST /api/login        → 接 ACCESS_PASSWORD 登录
"""

from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from server.auth import _make_token, hash_password, is_hash, verify_password
from server.services import settings_service

router = APIRouter()


class SettingsPayload(BaseModel):
    backend: str = Field(default="claude_cli")
    access_password: str = Field(default="")
    claude_cli: dict[str, Any] = Field(default_factory=dict)
    openai_api: dict[str, Any] = Field(default_factory=dict)
    chat_defaults: dict[str, Any] = Field(default_factory=dict)

    @field_validator("backend")
    @classmethod
    def backend_value(cls, v: str) -> str:
        v = (v or "").strip().lower()
        if v not in {"claude_cli", "openai_api"}:
            raise ValueError("backend must be 'claude_cli' or 'openai_api'")
        return v

    @field_validator("access_password")
    @classmethod
    def password_length(cls, v: str) -> str:
        if len(v) > 256:
            raise ValueError("access_password too long")
        return v


@router.get("/settings")
async def get_settings():
    return settings_service.public_view()


@router.get("/settings/check")
async def check_settings():
    return {
        "is_configured": settings_service.is_configured(),
        "backend": settings_service.active_backend_name(),
        "claude_cli_available": bool(settings_service.public_view()["claude_cli_available"]),
    }


@router.put("/settings")
async def update_settings(payload: SettingsPayload):
    """完整覆盖写入。前端必须 GET 后修改再 PUT，不能用 PATCH。

    API key 在 PUT 时必须传入完整值（GET 时是脱敏后空字符串），
    所以前端如果想保留旧 key 必须在 PUT 时手动合并。
    简单方案：前端 GET 时不显示已存在的 key，只显示占位「****」；用户重新填写才提交。
    """
    body = payload.model_dump()
    current = settings_service.read()
    # 如果前端在某个模型 profile 里保留 ****，用旧值兜底
    new_openai = body.get("openai_api") or {}
    new_profiles = new_openai.get("models") or []
    old_profiles = (current.get("openai_api") or {}).get("models") or []
    by_key = {p.get("key"): p for p in old_profiles}
    for p in new_profiles:
        key_val = p.get("api_key", "")
        if (not key_val) or key_val == "****":
            old = by_key.get(p.get("key"))
            if old:
                p["api_key"] = old.get("api_key", "")
    # 访问密码：
    # - "__KEEP__"   → 保留原值（哈希）
    # - ""           → 清空密码（不启用认证）
    # - 其它字符串    → 视为新明文，哈希后存盘
    pwd_in = body.get("access_password")
    if pwd_in == "__KEEP__":
        body["access_password"] = current.get("access_password", "")
    elif pwd_in and not is_hash(pwd_in):
        body["access_password"] = hash_password(pwd_in)
    try:
        settings_service.write(body)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return settings_service.public_view()


# ── ACCESS_PASSWORD 登录（合并自老的 routes/auth.py）──

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
    if verify_password(req.password):
        # token 派生材料 = 当前哈希字符串本身
        token = _make_token(settings_service.access_password_hash())
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
