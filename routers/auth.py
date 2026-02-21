"""认证路由：登录、登出"""
import hmac

from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse

from config import ACCESS_PASSWORD, SESSION_TOKEN
from app_common import templates

router = APIRouter(tags=["auth"])


def setup_auth_middleware(app):
    """注册认证中间件到 app"""
    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        if not ACCESS_PASSWORD:
            return await call_next(request)
        path = request.url.path
        if path in ("/login", "/favicon.ico", "/api/scan-status"):
            return await call_next(request)
        token = request.cookies.get("fp_session")
        if not token or not hmac.compare_digest(token, SESSION_TOKEN):
            return RedirectResponse(url="/login", status_code=302)
        return await call_next(request)


@router.get("/login")
async def login_page(request: Request):
    """显示登录页面"""
    if not ACCESS_PASSWORD:
        return RedirectResponse(url="/", status_code=302)
    return templates.TemplateResponse("login.html", {"request": request, "error": ""})


@router.post("/login")
async def login_submit(request: Request):
    """验证密码并设置 session cookie"""
    form = await request.form()
    password = (form.get("password") or "").strip()
    if hmac.compare_digest(password, ACCESS_PASSWORD):
        response = RedirectResponse(url="/", status_code=302)
        response.set_cookie(key="fp_session", value=SESSION_TOKEN, httponly=True, samesite="lax")
        return response
    return templates.TemplateResponse("login.html", {"request": request, "error": "密码错误，请重试"})


@router.get("/logout")
async def logout():
    """登出：清除 session cookie"""
    response = RedirectResponse(url="/login", status_code=302)
    response.delete_cookie(key="fp_session")
    return response
