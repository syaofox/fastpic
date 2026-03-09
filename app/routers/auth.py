"""认证路由：登录、登出"""

import hmac

from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse

from app.app_common import templates
from app.config import ACCESS_PASSWORD, SESSION_TOKEN

router = APIRouter(tags=["auth"])

EXCLUDED_PATHS = {
    "/login",
    "/favicon.ico",
    "/api/scan-status",
    "/api/task-status",
    "/api/task-status/clear",
    "/api/queue-status",
    "/ws",
}


def setup_auth_middleware(app):
    """注册认证中间件到 app"""

    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        if not ACCESS_PASSWORD:
            return await call_next(request)

        path = request.url.path
        if path in EXCLUDED_PATHS or path.startswith("/static/"):
            return await call_next(request)

        token = request.cookies.get("fp_session")
        if not token or not hmac.compare_digest(token, SESSION_TOKEN):
            return RedirectResponse(url="/login", status_code=302)

        try:
            return await call_next(request)
        except RuntimeError as e:
            if str(e) == "No response returned.":
                return
            raise


def setup_error_suppressor_middleware(app):
    """抑制 BaseHTTPMiddleware 的 No response returned 错误"""

    @app.middleware("http")
    async def suppress_error_middleware(request: Request, call_next):
        try:
            response = await call_next(request)
            return response
        except RuntimeError as e:
            if str(e) == "No response returned.":
                from fastapi.responses import Response

                return Response(status_code=204)
            raise
        except TypeError as e:
            if "NoneType" in str(e) and "not callable" in str(e):
                from fastapi.responses import Response

                return Response(status_code=204)
            raise

        path = request.url.path
        if path in EXCLUDED_PATHS or path.startswith("/static/"):
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
    return templates.TemplateResponse(request, "login.html", {"error": ""})


@router.post("/login")
async def login_submit(request: Request):
    """验证密码并设置 session cookie"""
    form = await request.form()
    password = (form.get("password") or "").strip()
    if hmac.compare_digest(password, ACCESS_PASSWORD):
        response = RedirectResponse(url="/", status_code=302)
        response.set_cookie(key="fp_session", value=SESSION_TOKEN, httponly=True, samesite="lax")
        return response
    return templates.TemplateResponse(request, "login.html", {"error": "密码错误，请重试"})


@router.get("/logout")
async def logout():
    """登出：清除 session cookie"""
    response = RedirectResponse(url="/login", status_code=302)
    response.delete_cookie(key="fp_session")
    return response
