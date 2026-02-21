"""API 请求/响应 Pydantic 模型"""
from pydantic import BaseModel


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
    ids: list[int]


class DeleteFoldersRequest(BaseModel):
    paths: list[str]


class MergeFoldersRequest(BaseModel):
    folder_a: str
    folder_b: str
    target: str = "auto"  # "folder_a" | "folder_b" | "auto"


class DownloadZipRequest(BaseModel):
    image_ids: list[int] = []
    folder_paths: list[str] = []


class MoveImagesRequest(BaseModel):
    ids: list[int]
    target_path: str


class MoveFoldersRequest(BaseModel):
    paths: list[str]
    target_path: str


class CreateFolderRequest(BaseModel):
    path: str
    name: str
