"""Image 记录创建辅助"""
from models import Image, natural_sort_key


def create_image_record(
    filename: str,
    relative_path: str,
    modified_at: float,
    file_size: int,
    width: int,
    height: int,
    media_type: str = "image",
) -> Image:
    """创建 Image 记录，统一 natural_sort_key 等字段"""
    return Image(
        filename=filename,
        relative_path=relative_path,
        modified_at=modified_at,
        file_size=file_size,
        width=width,
        height=height,
        filename_natural=natural_sort_key(filename),
        relative_path_natural=natural_sort_key(relative_path),
        media_type=media_type,
    )
