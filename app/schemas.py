"""API 请求/响应 Pydantic 模型"""

from enum import StrEnum
from typing import Generic, TypeVar

from pydantic import BaseModel, Field

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


# IN 子句分批上限，避免百万级 id 导致 max_allowed_packet 或查询退化
MAX_IDS_LENGTH = 2000


class AddTagsRequest(BaseModel):
    tags: list[str]


class RenameTagRequest(BaseModel):
    name: str


class MergeTagRequest(BaseModel):
    target: str


class BatchDeleteTagsRequest(BaseModel):
    names: list[str]


class ScanDuplicatesRequest(BaseModel):
    folder_path: str | None = None


class DeleteImagesRequest(BaseModel):
    ids: list[int] = Field(max_length=MAX_IDS_LENGTH)


class DeleteFoldersRequest(BaseModel):
    paths: list[str]


class RegenerateCoverRequest(BaseModel):
    paths: list[str]


class MergeFoldersRequest(BaseModel):
    folder_a: str
    folder_b: str
    target: str = "auto"  # "folder_a" | "folder_b" | "auto"
    duplicate_mode: str = "rename"  # "skip" | "rename" | "overwrite"


class DownloadZipRequest(BaseModel):
    image_ids: list[int] = Field(default_factory=list, max_length=MAX_IDS_LENGTH)
    folder_paths: list[str] = []


class MoveImagesRequest(BaseModel):
    ids: list[int] = Field(max_length=MAX_IDS_LENGTH)
    target_path: str


class MoveFoldersRequest(BaseModel):
    paths: list[str]
    target_path: str


class CreateFolderRequest(BaseModel):
    path: str
    name: str


class AddFolderThumbnailRequest(BaseModel):
    relative_path: str  # 图片相对路径，如 "2024/01/15/photo.jpg"，需在该文件夹下（含子目录）


class RenameFolderRequest(BaseModel):
    path: str  # 完整路径，如 "2024/01"
    new_name: str  # 新文件夹名（不含路径）


class RenameImageRequest(BaseModel):
    id: int
    new_filename: str


class BatchRenameInfoRequest(BaseModel):
    image_ids: list[int] = Field(default_factory=list, max_length=MAX_IDS_LENGTH)
    folder_paths: list[str] = []


class FolderRenameItem(BaseModel):
    path: str
    new_name: str


class ImageRenameItem(BaseModel):
    id: int
    new_filename: str


class BatchRenameRequest(BaseModel):
    folder_renames: list[FolderRenameItem] = []
    image_renames: list[ImageRenameItem] = []
