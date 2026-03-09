class ApiError extends Error {
    constructor(message, status, data) {
        super(message);
        this.name = 'ApiError';
        this.status = status;
        this.data = data;
    }
}

async function request(url, options = {}) {
    const defaultOptions = {
        headers: { 'Content-Type': 'application/json' },
    };
    
    const mergedOptions = {
        ...defaultOptions,
        ...options,
        headers: { ...defaultOptions.headers, ...options.headers },
    };
    
    const response = await fetch(url, mergedOptions);
    
    const contentType = response.headers.get('content-type');
    let data;
    if (contentType && contentType.includes('application/json')) {
        data = await response.json();
    } else {
        data = await response.text();
    }
    
    if (!response.ok) {
        const errorMessage = data?.detail || data?.message || `HTTP ${response.status}`;
        throw new ApiError(errorMessage, response.status, data);
    }
    
    return data;
}

export const api = {
    get(url, options) { return request(url, { ...options, method: 'GET' }); },
    post(url, body, options) { return request(url, { ...options, method: 'POST', body: JSON.stringify(body) }); },
    put(url, body, options) { return request(url, { ...options, method: 'PUT', body: JSON.stringify(body) }); },
    delete(url, options) { return request(url, { ...options, method: 'DELETE' }); },
    
    operations: {
        deleteImages: (ids) => api.post('/api/delete-images', { ids }),
        deleteFolders: (paths) => api.post('/api/delete-folders', { paths }),
        moveImages: (ids, targetPath) => api.post('/api/move-images', { ids, target_path: targetPath }),
        moveFolders: (paths, targetPath) => api.post('/api/move-folders', { paths, target_path: targetPath }),
        uploadFiles: async (files, path, options = {}) => {
            const formData = new FormData();
            formData.append('path', path);
            if (options.onDuplicate) formData.append('on_duplicate', options.onDuplicate);
            files.forEach(file => formData.append('files', file));
            const response = await fetch('/api/upload', { method: 'POST', body: formData });
            return response.json();
        },
        createFolder: (path, name) => api.post('/api/create-folder', { path, name }),
        renameFolder: (oldPath, newName) => api.put(`/api/rename-folder`, { path: oldPath, new_name: newName }),
        renameImage: (imageId, newName) => api.put(`/api/rename-image`, { id: imageId, new_filename: newName }),
        batchRename: (items) => api.post('/api/batch-rename', { items }),
        mergeFolder: (sourcePath, targetPath) => api.post('/api/merge-folders', { folder_a: sourcePath, folder_b: targetPath }),
        regenerateCovers: (paths) => api.post('/api/regenerate-covers', { paths }),
    },
    
    tasks: {
        scan: () => api.post('/scan', {}),
        cleanup: () => api.post('/api/cleanup', {}),
        fullSync: () => api.post('/api/full-sync', {}),
        scanDuplicates: (folderPath) => api.post('/api/scan-duplicates', { folder_path: folderPath }),
    },
};
