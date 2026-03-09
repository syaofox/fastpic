from enum import StrEnum
from typing import Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class ResponseStatus(StrEnum):
    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"


class ApiResponse(BaseModel, Generic[T]):
    status: ResponseStatus = ResponseStatus.SUCCESS
    message: str = ""
    data: T | None = None
    affected: list[int] = []
    errors: list[str] = []

    @classmethod
    def success(cls, data: T = None, message: str = "", affected: list[int] = None):
        return cls(status=ResponseStatus.SUCCESS, message=message, data=data, affected=affected or [])

    @classmethod
    def error(cls, message: str, errors: list[str] = None):
        return cls(status=ResponseStatus.ERROR, message=message, errors=errors or [])

    @classmethod
    def partial(cls, message: str, data: T = None, affected: list[int] = None, errors: list[str] = None):
        return cls(
            status=ResponseStatus.PARTIAL, message=message, data=data, affected=affected or [], errors=errors or []
        )
