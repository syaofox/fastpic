import { api } from './api.js';
import { galleryStore } from '../state/stores/galleryStore.js';
import { selectionStore } from '../state/stores/selectionStore.js';

class OperationService {
    generateOpId() {
        return `op_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }
    
    _extractData(result) {
        if (result && result.status) {
            if (result.status === 'success' || result.status === 'partial') {
                return result.data || {};
            }
        }
        return result || {};
    }
    
    async execute(operationType, apiCall, options = {}) {
        const opId = this.generateOpId();
        const { onSuccess, onError, successMessage, errorMessage } = options;
        
        try {
            const result = await apiCall();
            const data = this._extractData(result);
            
            if (onSuccess) onSuccess(data);
            
            if (successMessage !== false && typeof window.showToast === 'function') {
                window.showToast(successMessage || this.getDefaultMessage(operationType, data), 'success');
            }
            
            setTimeout(() => galleryStore.invalidateCache(), 500);
            return data;
            
        } catch (error) {
            if (onError) onError(error);
            if (typeof window.showToast === 'function') {
                window.showToast(errorMessage || error.message || '操作失败', 'error', 6000);
            }
            throw error;
        }
    }
    
    getDefaultMessage(operationType, result) {
        const messages = {
            'delete-images': `已删除 ${result.deleted} 项`,
            'delete-folders': `已删除 ${result.deleted_folders || result.deleted_images || 0} 个文件夹`,
            'move-images': `已移动 ${result.moved} 项`,
            'move-folders': `已移动 ${result.moved} 个文件夹`,
            'upload': `已上传 ${result.uploaded} 项${result.skipped ? `，${result.skipped} 项跳过` : ''}`,
            'create-folder': '文件夹已创建',
            'rename-folder': '文件夹已重命名',
            'merge-folder': '文件夹已合并',
            'batch-rename': `已重命名 ${result.image_count || result.folder_count || 0} 项`,
        };
        return messages[operationType] || '操作完成';
    }
    
    deleteImages(ids) {
        return this.execute('delete-images', () => api.operations.deleteImages(ids));
    }
    
    deleteFolders(paths) {
        return this.execute('delete-folders', () => api.operations.deleteFolders(paths));
    }
    
    moveImages(ids, targetPath) {
        return this.execute('move-images', () => api.operations.moveImages(ids, targetPath));
    }
    
    moveFolders(paths, targetPath) {
        return this.execute('move-folders', () => api.operations.moveFolders(paths, targetPath));
    }
    
    uploadFiles(files, path, options) {
        return this.execute('upload', () => api.operations.uploadFiles(files, path, options));
    }
    
    createFolder(path, name) {
        return this.execute('create-folder', () => api.operations.createFolder(path, name));
    }
    
    renameFolder(oldPath, newName) {
        return this.execute('rename-folder', () => api.operations.renameFolder(oldPath, newName));
    }
    
    batchRename(items) {
        return this.execute('batch-rename', () => api.operations.batchRename(items));
    }
    
    mergeFolder(sourcePath, targetPath) {
        return this.execute('merge-folder', () => api.operations.mergeFolder(sourcePath, targetPath));
    }
    
    async batchDelete(imageIds, folderPaths) {
        const promises = [];
        if (imageIds?.length) promises.push(this.deleteImages(imageIds));
        if (folderPaths?.length) promises.push(this.deleteFolders(folderPaths));
        const results = await Promise.all(promises);
        const totalDeleted = results.reduce((sum, r) => sum + (r.deleted || r.deleted_folders || r.deleted_images || 0), 0);
        selectionStore.clearSelection();
        return { deleted: totalDeleted };
    }
}

export const operationService = new OperationService();
