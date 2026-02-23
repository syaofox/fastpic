"""API 请求/响应 Pydantic 模型"""
from pydantic import BaseModel, Field

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


class MergeFoldersRequest(BaseModel):
    folder_a: str
    folder_b: str
    target: str = "auto"  # "folder_a" | "folder_b" | "auto"


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
