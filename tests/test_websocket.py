"""WebSocket 端点测试。"""

import pytest
from fastapi.testclient import TestClient


class TestWebSocketEndpoint:
    """测试 WebSocket 端点"""

    def test_websocket_endpoint_registered(self):
        """验证 WebSocket 端点已注册"""
        from app.main import app

        routes = [r.path for r in app.routes]
        assert "/ws" in routes

    def test_websocket_connection(self):
        """测试 WebSocket 连接"""
        from app.main import app

        client = TestClient(app)
        with client.websocket_connect("/ws") as ws:
            data = ws.receive_json()
            assert data["type"] == "connected"
            assert "payload" in data
            assert "client_id" in data["payload"]

    def test_websocket_ping_pong(self):
        """测试 ping/pong 消息"""
        from app.main import app

        client = TestClient(app)
        with client.websocket_connect("/ws") as ws:
            # 先接收连接消息
            data = ws.receive_json()
            assert data["type"] == "connected"

            # 发送 ping
            ws.send_json({"type": "ping", "payload": {}})
            data = ws.receive_json()
            assert data["type"] == "pong"
