"""测试 fixtures。"""

import os

import pytest

os.environ.setdefault("MYSQL_HOST", "127.0.0.1")


@pytest.fixture(autouse=True)
def setup_test_env(monkeypatch):
    """自动设置测试环境变量"""
    monkeypatch.setenv("MYSQL_HOST", "127.0.0.1")
