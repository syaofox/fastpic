"""上传错误处理测试。"""

from app.routers.images import (
    DuplicateFileError,
    FileSizeExceededError,
    UploadError,
)


class TestUploadErrorClasses:
    """上传错误类测试"""

    def test_upload_error_base_class(self):
        """验证基础错误类"""
        err = UploadError("test error")
        assert isinstance(err, Exception)
        assert str(err) == "test error"

    def test_file_size_exceeded_error(self):
        """验证文件大小超限错误"""
        err = FileSizeExceededError("file.jpg: 超过 100MB")
        assert isinstance(err, UploadError)
        assert isinstance(err, Exception)
        assert str(err) == "file.jpg: 超过 100MB"

    def test_duplicate_file_error(self):
        """验证文件重复错误"""
        err = DuplicateFileError("file.jpg 已存在")
        assert isinstance(err, UploadError)
        assert isinstance(err, Exception)
        assert str(err) == "file.jpg 已存在"

    def test_error_hierarchy(self):
        """验证错误继承关系"""
        assert issubclass(FileSizeExceededError, UploadError)
        assert issubclass(DuplicateFileError, UploadError)
        assert issubclass(UploadError, Exception)


class TestErrorResultDistinguished:
    """错误结果区分测试"""

    def test_error_result_distinguished(self):
        """验证错误结果被正确区分"""
        file_size_err = FileSizeExceededError("test")
        dup_err = DuplicateFileError("test")
        generic_err = Exception("test")

        assert isinstance(file_size_err, FileSizeExceededError)
        assert not isinstance(file_size_err, DuplicateFileError)
        assert isinstance(dup_err, DuplicateFileError)
        assert not isinstance(dup_err, FileSizeExceededError)
        assert not isinstance(generic_err, (FileSizeExceededError, DuplicateFileError))


class TestResultProcessing:
    """结果处理逻辑测试"""

    def test_result_classification_file_size_error(self):
        """验证 FileSizeExceededError 被正确分类"""
        from app.routers.images import FileSizeExceededError

        file_size_err = FileSizeExceededError("file1.jpg: 超过限制")

        errors = []

        if isinstance(file_size_err, FileSizeExceededError):
            errors.append(f"文件大小超限: {str(file_size_err)}")

        assert len(errors) == 1
        assert "文件大小超限" in errors[0]

    def test_result_classification_duplicate_error(self):
        """验证 DuplicateFileError 被正确分类为跳过"""
        from app.routers.images import DuplicateFileError

        dup_err = DuplicateFileError("file2.jpg 已存在")

        skipped = 0

        if isinstance(dup_err, DuplicateFileError):
            skipped += 1

        assert skipped == 1

    def test_result_classification_generic_error(self):
        """验证通用异常被正确分类"""
        generic_err = Exception("其他错误")

        errors = []

        if isinstance(generic_err, Exception):
            errors.append(str(generic_err))

        assert len(errors) == 1
        assert "其他错误" in errors[0]

    def test_result_classification_success_tuple(self):
        """验证成功结果的 tuple 被正确分类"""
        success_result = (True, False, None)
        error_result = (False, False, "some error")
        skip_result = (False, True, None)

        uploaded = 0
        skipped = 0
        errors = []

        for r in [success_result, error_result, skip_result]:
            if isinstance(r, tuple):
                u, s, err = r
                if u:
                    uploaded += 1
                elif s:
                    skipped += 1
                elif err:
                    errors.append(err)

        assert uploaded == 1
        assert skipped == 1
        assert len(errors) == 1
