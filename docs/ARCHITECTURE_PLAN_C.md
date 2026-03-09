# FastPic 前后端交互架构重构方案

## 概述

本方案对现有前后端交互模式进行完全重构，引入 WebSocket 进行实时通信、统一的状态管理和操作队列，旨在解决当前存在的通知不及时、刷新逻辑分散、API 响应格式不统一等问题。

**本方案为一次性完全迁移，不保留旧架构。迁移完成后删除所有 SSE 相关代码。**

---

## 1. 技术选型

### 1.1 WebSocket 框架

使用 FastAPI 内置 WebSocket 支持，不额外引入框架。

```python
from fastapi import WebSocket, WebSocketDisconnect

class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, set[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        if client_id not in self.active_connections:
            self.active_connections[client_id] = set()
        self.active_connections[client_id].add(websocket)
    
    def disconnect(self, websocket: WebSocket, client_id: str):
        if client_id in self.active_connections:
            self.active_connections[client_id].discard(websocket)
            if not self.active_connections[client_id]:
                del self.active_connections[client_id]
    
    async def send_personal(self, message: dict, client_id: str):
        if client_id in self.active_connections:
            for connection in self.active_connections[client_id]:
                await connection.send_json(message)
    
    async def broadcast(self, message: dict):
        for connections in self.active_connections.values():
            for connection in connections:
                await connection.send_json(message)
```

### 1.2 前端状态管理

自实现简化版 Signals 系统。

```javascript
class Signal {
    constructor(initialValue) {
        this._value = initialValue;
        this._callbacks = new Set();
    }
    
    get value() {
        return this._value;
    }
    
    set value(newValue) {
        if (this._value !== newValue) {
            this._value = newValue;
            this._notify();
        }
    }
    
    subscribe(callback) {
        this._callbacks.add(callback);
        return () => this._callbacks.delete(callback);
    }
    
    _notify() {
        this._callbacks.forEach(cb => cb(this._value));
    }
}

function computed(computeFn) {
    const signal = new Signal(computeFn());
    let lastValue = signal.value;
    
    const tracker = new Signal(0);
    tracker.subscribe(() => {
        const newValue = computeFn();
        if (newValue !== lastValue) {
            lastValue = newValue;
            signal._value = newValue;
            signal._notify();
        }
    });
    
    return {
        get value() { return signal.value; },
        subscribe: signal.subscribe.bind(signal),
    };
}
```

### 1.3 消息类型定义

```python
class MessageType(str, Enum):
    TASK_START = "task_start"
    TASK_PROGRESS = "task_progress"
    TASK_COMPLETE = "task_complete"
    TASK_ERROR = "task_error"
    GALLERY_UPDATE = "gallery_update"
    NOTIFICATION = "notification"
    SCAN_STATUS = "scan_status"
```

---

## 2. 架构设计

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                         Browser                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │  HTMX       │  │  WebSocket  │  │  Global State (Signals)  │ │
│  │  页面交互   │  │  实时通知   │  │  UI 状态同步            │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└────────────────────────────┬────────────────────────────────────┘
                             │
                    HTTP / WebSocket
                             │
┌────────────────────────────┴────────────────────────────────────┐
│                         Server                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  FastAPI    │  │  WebSocket  │  │  Message Broadcaster   │  │
│  │  REST API   │  │  Manager    │  │  任务广播              │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  Task       │  │  Operation  │  │  Task Queue           │  │
│  │  Manager    │  │  Service    │  │  后台任务             │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 消息流图

```
操作触发
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                     Frontend                                 │
│  1. 发送 HTTP 请求（操作）                                  │
│  2. 等待响应                                                │
└────────────────────────────┬──────────────────────────────────┘
                             │
                    HTTP Response
                             │
    ┌─────────────────────────┼─────────────────────────┐
    │                         │                         │
    ▼                         ▼                         ▼
 成功                      部分成功                  失败
    │                         │                         │
    ▼                         ▼                         ▼
广播消息给 WS          广播消息给 WS           显示错误 Toast
    │                         │                         │
    └─────────────────────────┼─────────────────────────┘
                              │
                              ▼
                  ┌───────────────────────────┐
                  │  WebSocket Broadcast     │
                  │   (task_state 更新)      │
                  └────────────┬──────────────┘
                               │
                               ▼
                  ┌───────────────────────────┐
                  │  Global State 更新        │
                  │  + Toast 通知            │
                  │  + 区域刷新              │
                  └───────────────────────────┘
```

---

## 3. 前端设计

### 3.1 目录结构

```
app/static/js/
├── state/
│   ├── index.js              # 状态导出入口
│   ├── signals.js             # 信号实现
│   └── stores/
│       ├── taskStore.js       # 任务状态
│       ├── galleryStore.js    # 画廊状态
│       ├── uiStore.js         # UI 状态
│       └── selectionStore.js  # 选中状态
├── services/
│   ├── websocket.js           # WebSocket 管理
│   ├── api.js                 # 统一 API 调用
│   └── operations.js         # 操作服务
├── components/
│   ├── Toast.js               # Toast 组件（替换 base.html 中的内联代码）
│   └── Progress.js            # 进度条组件
└── main.js                    # 入口文件
```

### 3.2 状态管理设计

```javascript
// stores/taskStore.js
export const taskStore = {
    _currentTask: new Signal(null),
    _queue: new Signal([]),
    
    get isRunning() { return this._currentTask.value !== null; },
    get progress() { return this._currentTask.value?.progress_percent ?? 0; },
    get currentOperation() { return this._currentTask.value?.current_operation ?? ''; },
    
    startTask(taskType, title, totalItems = 0) {
        this._currentTask.value = {
            task_type: taskType,
            title: title,
            total_items: totalItems,
            processed_items: 0,
            progress_percent: 0,
            started_at: new Date().toISOString(),
        };
    },
    
    updateProgress(processed, total, operation) {
        const current = this._currentTask.value;
        if (current) {
            this._currentTask.value = {
                ...current,
                processed_items: processed,
                progress_percent: total > 0 ? Math.round((processed / total) * 100) : 0,
                current_operation: operation,
            };
        }
    },
    
    complete(result) {
        const current = this._currentTask.value;
        if (current) {
            this._currentTask.value = {
                ...current,
                finished_at: new Date().toISOString(),
                result: result,
            };
        }
    },
    
    fail(error) {
        const current = this._currentTask.value;
        if (current) {
            this._currentTask.value = {
                ...current,
                finished_at: new Date().toISOString(),
                error: error,
            };
        }
    },
    
    clear() { this._currentTask.value = null; },
    
    subscribe(callback) { return this._currentTask.subscribe(callback); },
};
```

```javascript
// stores/galleryStore.js
export const galleryStore = {
    currentPath: new Signal(''),
    viewMode: new Signal('folder'),
    sortBy: new Signal('modified_at'),
    sortOrder: new Signal('desc'),
    filters: new Signal({
        filename: '', sizeMin: '', sizeMax: '',
        dateFrom: '', dateTo: '', tag: '',
    }),
    page: new Signal(1),
    hasNext: new Signal(false),
    _cache: new Map(),
    
    setPath(path) {
        this.currentPath.value = path;
        this.page.value = 1;
        this._cache.clear();
    },
    
    setViewMode(mode) { this.viewMode.value = mode; },
    
    updateFilters(newFilters) {
        this.filters.value = { ...this.filters.value, ...newFilters };
        this.page.value = 1;
    },
    
    invalidateCache(path = '') {
        if (path) {
            for (const key of this._cache.keys()) {
                if (key.startsWith(path)) this._cache.delete(key);
            }
        } else {
            this._cache.clear();
        }
    },
};
```

```javascript
// stores/selectionStore.js
export const selectionStore = {
    _selectedImages: new Set(),
    _selectedFolders: new Set(),
    _selectMode: new Signal(false),
    
    get selected this._selectedImagesImages() { return; },
    get selectedFolders() { return this._selectedFolders; },
    get selectMode() { return this._selectMode.value; },
    
    toggleImage(id) {
        if (this._selectedImages.has(id)) {
            this._selectedImages.delete(id);
        } else {
            this._selectedImages.add(id);
        }
    },
    
    toggleFolder(path) {
        if (this._selectedFolders.has(path)) {
            this._selectedFolders.delete(path);
        } else {
            this._selectedFolders.add(path);
        }
    },
    
    clearSelection() {
        this._selectedImages.clear();
        this._selectedFolders.clear();
        this._selectMode.value = false;
    },
    
    setSelectMode(enabled) { this._selectMode.value = enabled; },
};
```

### 3.3 WebSocket 服务

```javascript
// services/websocket.js
class WebSocketService {
    constructor() {
        this.ws = null;
        this.reconnectInterval = 3000;
        this.maxReconnectAttempts = 10;
        this.reconnectAttempts = 0;
        this.shouldReconnect = true;
    }
    
    connect() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        
        this.ws = new WebSocket(wsUrl);
        
        this.ws.onopen = () => {
            console.log('WebSocket connected');
            this.reconnectAttempts = 0;
        };
        
        this.ws.onmessage = (event) => {
            try {
                const message = JSON.parse(event.data);
                this.handleMessage(message);
            } catch (e) {
                console.error('Failed to parse WS message:', e);
            }
        };
        
        this.ws.onclose = () => {
            console.log('WebSocket disconnected');
            if (this.shouldReconnect && this.reconnectAttempts < this.maxReconnectAttempts) {
                this.reconnectAttempts++;
                setTimeout(() => this.connect(), this.reconnectInterval);
            }
        };
        
        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
    }
    
    disconnect() {
        this.shouldReconnect = false;
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
    }
    
    handleMessage(message) {
        const { type, payload } = message;
        
        switch (type) {
            case 'task_start':
                taskStore.startTask(payload.task_type, payload.title, payload.total_items);
                break;
                
            case 'task_progress':
                taskStore.updateProgress(
                    payload.processed_items,
                    payload.total_items,
                    payload.current_operation
                );
                break;
                
            case 'task_complete':
                taskStore.complete(payload.result);
                galleryStore.invalidateCache();
                if (payload.result_message) {
                    showToast(payload.result_message, 'success');
                }
                break;
                
            case 'task_error':
                taskStore.fail(payload.error);
                showToast(payload.error, 'error', 6000);
                break;
                
            case 'gallery_update':
                galleryStore.invalidateCache(payload.affected_path);
                break;
                
            case 'scan_status':
                this.handleScanStatus(payload);
                break;
                
            case 'notification':
                showToast(payload.message, payload.level || 'info');
                break;
                
            default:
                console.log('Unknown message type:', type);
        }
    }
    
    handleScanStatus(payload) {
        const banner = document.getElementById('scan-banner');
        if (banner) {
            if (payload.scanning) {
                banner.classList.remove('hidden');
            } else {
                banner.classList.add('hidden');
            }
        }
    }
    
    send(type, payload) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type, payload }));
        }
    }
}

export const wsService = new WebSocketService();
```

### 3.4 统一 API 服务

```javascript
// services/api.js
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
        createFolder: (path, name) => api.post('/api/folders', { path, name }),
        renameFolder: (oldPath, newName) => api.put(`/api/folders/${encodeURIComponent(oldPath)}`, { name: newName }),
        renameImage: (imageId, newName) => api.put(`/api/images/${imageId}/rename`, { name: newName }),
        batchRename: (items) => api.post('/api/batch-rename', { items }),
        mergeFolder: (sourcePath, targetPath) => api.post('/api/folders/merge', { source: sourcePath, target: targetPath }),
    },
    
    tasks: {
        scan: () => api.post('/scan', {}),
        cleanup: () => api.post('/api/cleanup', {}),
        fullSync: () => api.post('/api/full-sync', {}),
        scanDuplicates: (folderPath) => api.post('/api/scan-duplicates', { folder_path: folderPath }),
    },
};
```

### 3.5 操作服务

```javascript
// services/operations.js
import { api } from './api.js';
import { galleryStore } from '../state/stores/galleryStore.js';
import { selectionStore } from '../state/stores/selectionStore.js';
import { showToast } from '../components/Toast.js';

class OperationService {
    generateOpId() {
        return `op_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }
    
    async execute(operationType, apiCall, options = {}) {
        const opId = this.generateOpId();
        const { onSuccess, onError, successMessage, errorMessage } = options;
        
        try {
            const result = await apiCall();
            
            if (onSuccess) onSuccess(result);
            
            if (successMessage !== false) {
                showToast(successMessage || this.getDefaultMessage(operationType, result), 'success');
            }
            
            setTimeout(() => galleryStore.invalidateCache(), 500);
            return result;
            
        } catch (error) {
            if (onError) onError(error);
            showToast(errorMessage || error.message || '操作失败', 'error', 6000);
            throw error;
        }
    }
    
    getDefaultMessage(operationType, result) {
        const messages = {
            'delete-images': `已删除 ${result.deleted} 项`,
            'delete-folders': `已删除 ${result.deleted_folders} 个文件夹`,
            'move-images': `已移动 ${result.moved} 项`,
            'move-folders': `已移动 ${result.moved} 个文件夹`,
            'upload': `已上传 ${result.uploaded} 项${result.skipped ? `，${result.skipped} 项跳过` : ''}`,
            'create-folder': '文件夹已创建',
            'rename-folder': '文件夹已重命名',
            'merge-folder': '文件夹已合并',
            'batch-rename': `已重命名 ${result.renamed} 项`,
        };
        return messages[operationType] || '操作完成';
    }
    
    // 便捷方法
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
    
    // 批量操作
    async batchDelete(imageIds, folderPaths) {
        const promises = [];
        if (imageIds?.length) promises.push(this.deleteImages(imageIds));
        if (folderPaths?.length) promises.push(this.deleteFolders(folderPaths));
        const results = await Promise.all(promises);
        const totalDeleted = results.reduce((sum, r) => sum + (r.deleted || r.deleted_folders || 0), 0);
        selectionStore.clearSelection();
        return { deleted: totalDeleted };
    }
}

export const operationService = new OperationService();
```

### 3.6 Toast 组件（替换 base.html 中的内联代码）

```javascript
// components/Toast.js
class ToastContainer {
    constructor() {
        this.container = null;
        this.init();
    }
    
    init() {
        if (document.getElementById('toast-container')) return;
        this.container = document.createElement('div');
        this.container.id = 'toast-container';
        this.container.className = 'fixed bottom-4 left-1/2 -translate-x-1/2 z-[120] flex flex-col gap-2 pointer-events-none';
        this.container.setAttribute('aria-live', 'polite');
        document.body.appendChild(this.container);
    }
    
    show(message, type = 'info', duration = 4000) {
        const config = {
            success: { classes: 'bg-green-100 text-green-800 border border-green-200', icon: '<svg class="w-4 h-4 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg>' },
            error: { classes: 'bg-red-100 text-red-800 border border-red-200', icon: '<svg class="w-4 h-4 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path></svg>' },
            warning: { classes: 'bg-amber-100 text-amber-800 border border-amber-200', icon: '<svg class="w-4 h-4 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path></svg>' },
            info: { classes: 'bg-slate-800 text-white', icon: '<svg class="w-4 h-4 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>' },
        };
        
        const conf = config[type] || config.info;
        const el = document.createElement('div');
        el.className = `px-4 py-3 rounded-lg shadow-lg text-sm font-medium max-w-md flex items-center gap-2.5 ${conf.classes}`;
        el.innerHTML = `${conf.icon}<span>${message}</span>`;
        
        this.container.appendChild(el);
        
        if (duration > 0) {
            setTimeout(() => {
                el.style.opacity = '0';
                el.style.transition = 'opacity 0.2s';
                setTimeout(() => el.remove(), 200);
            }, duration);
        }
    }
}

const toastContainer = new ToastContainer();
export function showToast(message, type = 'info', duration = 4000) {
    toastContainer.show(message, type, duration);
}
```

### 3.7 进度指示器组件

```javascript
// components/Progress.js
class ProgressIndicator {
    constructor() {
        this.element = null;
        this.init();
    }
    
    init() {
        this.element = document.getElementById('progress-indicator');
        if (!this.element) return;
        
        this.titleEl = document.getElementById('progress-title');
        this.percentEl = document.getElementById('progress-percent');
        this.barEl = document.getElementById('progress-bar');
        this.messageEl = document.getElementById('progress-message');
        this.closeBtn = document.getElementById('progress-close');
        
        if (this.closeBtn) {
            this.closeBtn.onclick = () => this.hide();
        }
    }
    
    show(taskId, title, progress = 0, message = '') {
        this.init();
        if (!this.element) return;
        
        if (this.titleEl) this.titleEl.textContent = title || '正在处理...';
        if (this.percentEl) this.percentEl.textContent = Math.round(progress) + '%';
        if (this.barEl) this.barEl.style.width = progress + '%';
        if (this.messageEl) this.messageEl.textContent = message || '';
        
        this.element.classList.remove('hidden');
        requestAnimationFrame(() => {
            this.element.classList.remove('-translate-y-full');
        });
    }
    
    update(progress, message) {
        if (!this.element || this.element.classList.contains('hidden')) return;
        
        if (this.percentEl) this.percentEl.textContent = Math.round(progress) + '%';
        if (this.barEl) this.barEl.style.width = progress + '%';
        if (this.messageEl && message !== undefined) this.messageEl.textContent = message;
    }
    
    hide() {
        if (!this.element) return;
        
        this.element.classList.add('-translate-y-full');
        setTimeout(() => {
            if (this.element.classList.contains('-translate-y-full')) {
                this.element.classList.add('hidden');
            }
        }, 300);
    }
}

const progressIndicator = new ProgressIndicator();
export { progressIndicator };
```

### 3.8 主入口文件

```javascript
// main.js
import { wsService } from './services/websocket.js';
import { taskStore } from './state/stores/taskStore.js';
import { galleryStore } from './state/stores/galleryStore.js';
import { selectionStore } from './state/stores/selectionStore.js';
import { showToast } from './components/Toast.js';
import { progressIndicator } from './components/Progress.js';

// 页面加载时连接 WebSocket
document.addEventListener('DOMContentLoaded', () => {
    wsService.connect();
    
    // 订阅任务状态变化，更新进度条
    taskStore.subscribe((task) => {
        if (task) {
            if (!task.finished_at) {
                progressIndicator.show(
                    task.task_type,
                    task.title,
                    task.progress_percent,
                    task.current_operation
                );
            } else {
                progressIndicator.hide();
            }
        } else {
            progressIndicator.hide();
        }
    });
});

// HTMX 页面切换后确保 WebSocket 连接正常
document.addEventListener('htmx:afterSwap', (event) => {
    if (event.detail.target.id === 'gallery-container' || 
        event.detail.target.id === 'main-content') {
        if (!wsService.ws || wsService.ws.readyState !== WebSocket.OPEN) {
            wsService.connect();
        }
    }
});

// 页面卸载时断开连接
window.addEventListener('beforeunload', () => {
    wsService.disconnect();
});

// 导出全局函数供 HTML 调用
window.galleryStore = galleryStore;
window.selectionStore = selectionStore;
window.showToast = showToast;
window.wsService = wsService;
```

---

## 4. 后端设计

### 4.1 WebSocket 端点

```python
# app/routers/websocket.py
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from datetime import datetime
import json
import uuid

router = APIRouter()


class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, set[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        if client_id not in self.active_connections:
            self.active_connections[client_id] = set()
        self.active_connections[client_id].add(websocket)
    
    def disconnect(self, websocket: WebSocket, client_id: str):
        if client_id in self.active_connections:
            self.active_connections[client_id].discard(websocket)
            if not self.active_connections[client_id]:
                del self.active_connections[client_id]
    
    async def broadcast(self, message: dict):
        for connections in self.active_connections.values():
            for connection in connections:
                try:
                    await connection.send_json(message)
                except Exception:
                    pass


manager = ConnectionManager()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    client_id = f"client_{id(websocket)}"
    await manager.connect(websocket, client_id)
    
    try:
        await websocket.send_json({
            "type": "connected",
            "payload": {"client_id": client_id},
            "timestamp": datetime.now().isoformat(),
        })
        
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                await handle_client_message(message, client_id)
            except json.JSONDecodeError:
                pass
                
    except WebSocketDisconnect:
        manager.disconnect(websocket, client_id)
    except Exception:
        manager.disconnect(websocket, client_id)


async def handle_client_message(message: dict, client_id: str):
    msg_type = message.get("type")
    payload = message.get("payload", {})
    
    if msg_type == "ping":
        await manager.broadcast({
            "type": "pong",
            "payload": {},
            "timestamp": datetime.now().isoformat(),
        })
```

### 4.2 消息广播服务

```python
# app/services/message_broadcaster.py
from datetime import datetime
from typing import Any
from enum import Enum
import uuid

class MessageType(str, Enum):
    TASK_START = "task_start"
    TASK_PROGRESS = "task_progress"
    TASK_COMPLETE = "task_complete"
    TASK_ERROR = "task_error"
    GALLERY_UPDATE = "gallery_update"
    NOTIFICATION = "notification"
    SCAN_STATUS = "scan_status"


class MessageBroadcaster:
    def __init__(self):
        from app.routers.websocket import manager
        self._manager = manager
    
    def _create_message(self, msg_type: MessageType, payload: dict[str, Any]) -> dict:
        return {
            "type": msg_type.value,
            "payload": payload,
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
        }
    
    async def broadcast_task_start(self, task_type: str, title: str, total_items: int = 0):
        message = self._create_message(
            MessageType.TASK_START,
            {"task_type": task_type, "title": title, "total_items": total_items}
        )
        await self._manager.broadcast(message)
    
    async def broadcast_task_progress(self, task_type: str, processed_items: int, total_items: int, current_operation: str = ""):
        message = self._create_message(
            MessageType.TASK_PROGRESS,
            {
                "task_type": task_type,
                "processed_items": processed_items,
                "total_items": total_items,
                "progress_percent": int((processed_items / total_items) * 100) if total_items > 0 else 0,
                "current_operation": current_operation,
            }
        )
        await self._manager.broadcast(message)
    
    async def broadcast_task_complete(self, task_type: str, result: dict[str, Any], result_message: str = ""):
        message = self._create_message(
            MessageType.TASK_COMPLETE,
            {"task_type": task_type, "result": result, "result_message": result_message}
        )
        await self._manager.broadcast(message)
    
    async def broadcast_task_error(self, task_type: str, error: str):
        message = self._create_message(
            MessageType.TASK_ERROR,
            {"task_type": task_type, "error": error}
        )
        await self._manager.broadcast(message)
    
    async def broadcast_gallery_update(self, affected_path: str = "", action: str = "update"):
        message = self._create_message(
            MessageType.GALLERY_UPDATE,
            {"affected_path": affected_path, "action": action}
        )
        await self._manager.broadcast(message)
    
    async def broadcast_scan_status(self, scanning: bool):
        message = self._create_message(
            MessageType.SCAN_STATUS,
            {"scanning": scanning}
        )
        await self._manager.broadcast(message)
    
    async def broadcast_notification(self, message: str, level: str = "info"):
        message = self._create_message(
            MessageType.NOTIFICATION,
            {"message": message, "level": level}
        )
        await self._manager.broadcast(message)


broadcaster = MessageBroadcaster()
```

### 4.3 统一 API 响应格式

```python
# app/schemas/response.py
from typing import Any, Generic, TypeVar, Optional
from pydantic import BaseModel
from enum import Enum

class ResponseStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"


class ApiResponse(BaseModel, Generic[T]):
    status: ResponseStatus = ResponseStatus.SUCCESS
    message: str = ""
    data: Optional[T] = None
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
        return cls(status=ResponseStatus.PARTIAL, message=message, data=data, affected=affected or [], errors=errors or [])
```

### 4.4 任务服务集成

```python
# app/services/task_service.py
from typing import Any, Callable, Awaitable
from dataclasses import dataclass
from app.services.message_broadcaster import broadcaster
from app.services import task_state

@dataclass
class TaskContext:
    task_type: str
    title: str
    total_items: int = 0
    processed_items: int = 0
    
    async def broadcast_start(self):
        await broadcaster.broadcast_task_start(self.task_type, self.title, self.total_items)
    
    async def broadcast_progress(self, processed: int, operation: str = ""):
        self.processed_items = processed
        await broadcaster.broadcast_task_progress(
            self.task_type, processed, self.total_items, operation
        )
    
    async def broadcast_complete(self, result: dict[str, Any], message: str = ""):
        await broadcaster.broadcast_task_complete(self.task_type, result, message)
    
    async def broadcast_error(self, error: str):
        await broadcaster.broadcast_task_error(self.task_type, error)


class TaskService:
    def __init__(self):
        self._handlers: dict[str, Callable[[TaskContext, Any], Awaitable[dict]]] = {}
    
    def register(self, task_type: str):
        def decorator(func: Callable[[TaskContext, Any], Awaitable[dict]]):
            self._handlers[task_type] = func
            return func
        return decorator
    
    async def execute(self, task_type: str, title: str, params: Any = None, total_items: int = 0) -> dict[str, Any]:
        if not task_state.start_task(task_type, total_items, title):
            return {"error": "有任务正在进行中"}
        
        context = TaskContext(task_type=task_type, title=title, total_items=total_items)
        
        try:
            await context.broadcast_start()
            
            handler = self._handlers.get(task_type)
            if not handler:
                raise ValueError(f"Unknown task type: {task_type}")
            
            result = await handler(context, params)
            
            result_message = self._format_result_message(task_type, result)
            await context.broadcast_complete(result, result_message)
            task_state.end_task(result)
            
            return result
            
        except Exception as e:
            error_msg = str(e)
            await context.broadcast_error(error_msg)
            task_state.fail_task(error_msg)
            return {"error": error_msg}
    
    def _format_result_message(self, task_type: str, result: dict[str, Any]) -> str:
        messages = {
            "scan": f"扫描完成，发现 {result.get('scanned', 0)} 个文件",
            "cleanup": f"清理完成，移除 {result.get('stale_removed', 0)} 条记录",
            "full-sync": f"同步完成，发现 {result.get('images_added', 0) + result.get('videos_added', 0)} 个新文件",
            "upload": f"已上传 {result.get('uploaded', 0)} 个文件",
            "delete-images": f"已删除 {result.get('deleted', 0)} 项",
            "delete-folders": f"已删除 {result.get('deleted_folders', 0)} 个文件夹",
            "move-images": f"已移动 {result.get('moved', 0)} 项",
            "move-folders": f"已移动 {result.get('moved', 0)} 个文件夹",
        }
        return messages.get(task_type, "操作完成")


task_service = TaskService()
```

---

## 5. 迁移清单

### 5.1 需要删除的旧代码

| 文件 | 删除内容 |
|------|----------|
| `app/routers/settings.py` | `/api/task-events` SSE 端点及相关代码 |
| `app/services/task_state.py` | SSE 相关代码，仅保留内存任务状态 |
| `app/templates/base.html` | `showToast` 函数重复定义、`connectTaskEvents` 函数、scan-banner polling 代码 |
| `app/templates/base.html` | 进度指示器相关内联 JS（保留 HTML 结构，由新组件接管） |

### 5.2 需要改造的路由

| 路由文件 | 改造内容 |
|----------|----------|
| `app/routers/images.py` | 使用 `task_service.execute()` 包装操作，广播进度，返回统一响应格式 |
| `app/routers/folders.py` | 使用 `task_service.execute()` 包装操作，广播进度，返回统一响应格式 |
| `app/routers/settings.py` | 移除 SSE，扫描状态通过 WebSocket 广播 |
| `app/routers/auth.py` | 移除 `/api/task-events` 白名单 |

### 5.3 需要改造的前端功能

| 功能 | 改造方式 |
|------|----------|
| 删除图片 | 调用 `operationService.deleteImages(ids)` |
| 批量删除 | 调用 `operationService.batchDelete(imageIds, folderPaths)` |
| 移动图片 | 调用 `operationService.moveImages(ids, targetPath)` |
| 移动文件夹 | 调用 `operationService.moveFolders(paths, targetPath)` |
| 上传文件 | 调用 `operationService.uploadFiles(files, path, options)` |
| 创建文件夹 | 调用 `operationService.createFolder(path, name)` |
| 重命名文件夹 | 调用 `operationService.renameFolder(oldPath, newName)` |
| 合并文件夹 | 调用 `operationService.mergeFolder(sourcePath, targetPath)` |
| 批量重命名 | 调用 `operationService.batchRename(items)` |
| 扫描/清理/同步 | 调用 `api.tasks.scan()` 等 |

---

## 6. 完整实施计划

### 步骤 1：创建基础设施（1-2 天）

1. 创建 `app/routers/websocket.py`
2. 创建 `app/services/message_broadcaster.py`
3. 创建 `app/schemas/response.py`
4. 创建 `app/services/task_service.py`
5. 在 `app/main.py` 中注册 WebSocket 路由

### 步骤 2：改造后端路由（2-3 天）

1. 改造 `app/routers/images.py` - 所有操作使用 task_service
2. 改造 `app/routers/folders.py` - 所有操作使用 task_service
3. 改造 `app/routers/settings.py` - 移除 SSE，使用 WebSocket 广播扫描状态
4. 改造 `app/routers/auth.py` - 移除 SSE 白名单

### 步骤 3：创建前端基础设施（1-2 天）

1. 创建 `app/static/js/state/signals.js`
2. 创建 `app/static/js/state/stores/taskStore.js`
3. 创建 `app/static/js/state/stores/galleryStore.js`
4. 创建 `app/static/js/state/stores/selectionStore.js`
5. 创建 `app/static/js/state/index.js`
6. 创建 `app/static/js/services/websocket.js`
7. 创建 `app/static/js/services/api.js`
8. 创建 `app/static/js/services/operations.js`
9. 创建 `app/static/js/components/Toast.js`
10. 创建 `app/static/js/components/Progress.js`
11. 创建 `app/static/js/main.js`

### 步骤 4：改造前端 HTML 和 JS（2-3 天）

1. 改造 `app/templates/base.html`：
   - 移除重复的 `showToast` 定义
   - 移除 `connectTaskEvents` 函数
   - 移除 scan-banner polling 代码
   - 引入新的 JS 模块
2. 改造 `app/static/js/gallery.js`：
   - 使用新的 `operationService` 替代内联 fetch
   - 使用 `selectionStore` 替代全局变量
   - 使用 `showToast` 替代内联 toast 函数
   - 使用 `progressIndicator` 替代内联进度条代码

### 步骤 5：测试和调优（1-2 天）

1. 测试所有操作流程
2. 测试 WebSocket 重连
3. 测试页面导航后状态保持
4. 性能优化

---

## 7. 功能覆盖清单

迁移完成后，以下功能必须保持不变：

| 功能模块 | 功能点 | 状态 |
|----------|--------|------|
| 图片浏览 | 文件夹/列表/瀑布流三种视图模式 | 需保持 |
| 图片浏览 | 无限滚动加载 | 需保持 |
| 图片浏览 | 大图预览模态框 | 需保持 |
| 图片浏览 | 幻灯片播放 | 需保持 |
| 图片操作 | 单图/批量删除 | 需保持 |
| 图片操作 | 单图/批量移动 | 需保持 |
| 图片操作 | 上传图片/视频 | 需保持 |
| 图片操作 | 下载单图/打包下载 | 需保持 |
| 文件夹操作 | 创建文件夹 | 需保持 |
| 文件夹操作 | 重命名文件夹 | 需保持 |
| 文件夹操作 | 删除文件夹 | 需保持 |
| 文件夹操作 | 移动文件夹 | 需保持 |
| 文件夹操作 | 合并文件夹 | 需保持 |
| 文件夹操作 | 批量重命名 | 需保持 |
| 标签管理 | 添加/删除标签 | 需保持 |
| 标签管理 | 重命名/合并标签 | 需保持 |
| 筛选排序 | 按文件名/大小/日期/标签筛选 | 需保持 |
| 筛选排序 | 按名称/时间/大小排序 | 需保持 |
| 设置页面 | 手动扫描 | 需保持 |
| 设置页面 | 数据库清理 | 需保持 |
| 设置页面 | 完整同步 | 需保持 |
| 设置页面 | 扫描重复文件 | 需保持 |
| 任务进度 | 实时进度显示 | 需保持 |
| 任务进度 | 任务完成通知 | 需保持 |
| 扫描提示 | 后台扫描提示条 | 需保持 |
| 用户认证 | 登录/登出 | 需保持 |

---

## 8. 文件清单

### 新增文件

```
app/
├── routers/
│   └── websocket.py              # WebSocket 端点
├── services/
│   ├── message_broadcaster.py    # 消息广播服务
│   └── task_service.py          # 任务执行服务
└── schemas/
    └── response.py                # 统一响应格式

app/static/js/
├── state/
│   ├── index.js                  # 状态导出入口
│   ├── signals.js                # 信号实现
│   └── stores/
│       ├── taskStore.js          # 任务状态
│       ├── galleryStore.js      # 画廊状态
│       ├── selectionStore.js    # 选中状态
│       └── uiStore.js           # UI 状态
├── services/
│   ├── websocket.js              # WebSocket 管理
│   ├── api.js                    # 统一 API 调用
│   └── operations.js             # 操作服务
├── components/
│   ├── Toast.js                  # Toast 组件
│   └── Progress.js               # 进度条组件
└── main.js                       # 入口文件
```

### 修改文件

```
app/main.py                       # 注册 WebSocket 路由
app/routers/auth.py               # 移除 SSE 白名单
app/routers/images.py             # 使用 task_service + 统一响应
app/routers/folders.py           # 使用 task_service + 统一响应
app/routers/settings.py          # 移除 SSE，使用 WS 广播
app/templates/base.html           # 移除重复代码，引入新模块
app/static/js/gallery.js          # 使用新架构
app/static/js/folder-browser.js   # 保持不变（独立组件）
app/static/js/utils.js            # 保持不变（工具函数）
```

### 删除文件

```
无（全部为改造，无删除）
```

---

## 9. 测试策略

### 9.1 后端测试

```python
# tests/test_websocket.py
import pytest
from fastapi.testclient import TestClient
from app.main import app

def test_websocket_connection():
    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        data = ws.receive_json()
        assert data["type"] == "connected"

# tests/test_task_service.py
@pytest.mark.asyncio
async def test_task_execute():
    from app.services.task_service import task_service
    
    @task_service.register("test-task")
    async def handler(context, params):
        await context.broadcast_progress(5, 10, "处理中")
        return {"result": "ok"}
    
    result = await task_service.execute("test-task", "测试任务", {}, 10)
    assert result.get("result") == "ok"
```

### 9.2 前端测试

```javascript
// tests/stores.test.js
import { test, describe } from 'vitest';
import { galleryStore } from '../app/static/js/state/stores/galleryStore.js';

describe('galleryStore', () => {
    test('setPath clears page', () => {
        galleryStore.page.value = 5;
        galleryStore.setPath('test/path');
        // 验证 page 重置为 1
    });
    
    test('invalidateCache clears cache', () => {
        galleryStore.setCachedData('path1', 1, {});
        galleryStore.setCachedData('path2', 1, {});
        galleryStore.invalidateCache();
        // 验证缓存已清空
    });
});
```

---

## 10. 注意事项

### 10.1 安全性

- WebSocket 连接需要与 HTTP 相同的认证机制
- 验证 WebSocket 消息来源

### 10.2 性能

- 限制最大连接数
- 消息频率控制（避免刷屏）
- 大批量操作时分批广播进度

### 10.3 兼容性

- 主流浏览器均支持 WebSocket
- 无需保留 HTTP 后备方案

---

## 11. 参考资料

- [FastAPI WebSocket](https://fastapi.tiangolo.com/advanced/websockets/)
- [Signals 模式](https://preactjs.com/guide/v10/signals/)
- [HTMX WebSocket](https://htmx.org/extensions/web-sockets/)
