"""ApiResponse schema 测试。"""

import pytest
from pydantic import ValidationError

from app.schemas import ApiResponse, ResponseStatus


class TestApiResponse:
    """测试 ApiResponse 模型"""

    def test_success_default(self):
        """默认创建返回成功状态"""
        response = ApiResponse()
        assert response.status == ResponseStatus.SUCCESS
        assert response.message == ""
        assert response.data is None
        assert response.affected == []
        assert response.errors == []

    def test_success_with_data(self):
        """成功响应带数据"""
        response = ApiResponse.success({"deleted": 5}, "删除成功")
        assert response.status == ResponseStatus.SUCCESS
        assert response.message == "删除成功"
        assert response.data == {"deleted": 5}
        assert response.affected == []

    def test_success_with_affected(self):
        """成功响应带受影响 ID 列表"""
        response = ApiResponse.success([1, 2, 3], affected=[1, 2, 3])
        assert response.status == ResponseStatus.SUCCESS
        assert response.data == [1, 2, 3]
        assert response.affected == [1, 2, 3]

    def test_error_response(self):
        """错误响应"""
        response = ApiResponse.error("操作失败", ["错误1", "错误2"])
        assert response.status == ResponseStatus.ERROR
        assert response.message == "操作失败"
        assert response.errors == ["错误1", "错误2"]

    def test_partial_response(self):
        """部分成功响应"""
        response = ApiResponse.partial("部分成功", {"moved": 3}, [1, 2, 3], ["部分错误"])
        assert response.status == ResponseStatus.PARTIAL
        assert response.message == "部分成功"
        assert response.data == {"moved": 3}
        assert response.affected == [1, 2, 3]
        assert response.errors == ["部分错误"]

    def test_response_status_enum(self):
        """验证 ResponseStatus 枚举值"""
        assert ResponseStatus.SUCCESS == "success"
        assert ResponseStatus.ERROR == "error"
        assert ResponseStatus.PARTIAL == "partial"


class TestApiResponseWithTypes:
    """测试 ApiResponse 泛型类型"""

    def test_with_dict_data(self):
        """带字典数据"""
        response = ApiResponse.success({"key": "value"})
        assert response.data == {"key": "value"}

    def test_with_list_data(self):
        """带列表数据"""
        response = ApiResponse.success([1, 2, 3])
        assert response.data == [1, 2, 3]

    def test_with_none_data(self):
        """data 为 None"""
        response = ApiResponse.success(None)
        assert response.data is None
