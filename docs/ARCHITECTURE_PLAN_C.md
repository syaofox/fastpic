# FastPic 前后端交互架构设计方案 C

## 概述

本方案对现有前后端交互模式进行较大重构，引入 WebSocket 进行实时通信、统一的状态管理和操作队列，旨在解决当前存在的通知不及时、刷新逻辑分散、API 响应格式不统一等问题。

**注意**：本方案为理想状态的设计，实际实现时可根据优先级分阶段进行。

---

## 1. 技术选型

### 1.1 WebSocket 框架

**方案**：使用 FastAPI 内置 WebSocket 支持，不额外引入框架

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

**方案**：使用轻量级信号（Signals）库，如 `signals.js` 或自实现简化版

```javascript
// 自实现简化信号系统
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

// 全局状态容器
const GlobalState = {
    // 任务状态
    taskState: new Signal(null),
    
    // 当前路径
    currentPath: new Signal(''),
    
    // 选中项
    selectedItems: new Signal(new Set()),
    
    // 模态框状态
    modalState: new Signal({ isOpen: false, images: [], index: 0 }),
    
    // Toast 队列
    toastQueue: new Signal([]),
};
```

### 1.3 消息队列（可选）

**方案**：如需要支持离线任务，使用内存队列；如只需实时通知，可省略

```python
from asyncio import Queue
from dataclasses import dataclass, field
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

@dataclass
class WSMessage:
    type: MessageType
    payload: dict[str, Any]
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

class MessageQueue:
    def __init__(self):
        self._queue: Queue[WSMessage] = Queue()
    
    async def put(self, message: WSMessage):
        await self._queue.put(message)
    
    async def get(self) -> WSMessage:
        return await self._queue.get()
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
│  │  FastAPI    │  │  WebSocket  │  │  Message Queue         │  │
│  │  REST API   │  │  Manager    │  │  任务广播              │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  Task       │  │  Operation  │  │  Cache                 │  │
│  │  Manager    │  │  Service    │  │  Manager               │  │
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
│  2. 立即更新本地状态（乐观更新）                            │
│  3. 等待响应                                                │
└────────────────────────────┬──────────────────────────────────┘
                             │
                    HTTP Response
                             │
    ┌─────────────────────────┼─────────────────────────┐
    │                         │                         │
    ▼                         ▼                         ▼
成功                  重试/回退              失败
    │                         │                         │
    ▼                         ▼                         ▼
发送确认消息         显示进度             显示错误 Toast
给 WS                  │                         │
    │                     ▼                         │
    │              发送确认消息                       │
    │              给 WS                             │
    │                     │                         │
    └─────────────────────┼─────────────────────────┘
                          │
                          ▼
              ┌───────────────────────────┐
              │     WebSocket Broadcast   │
              │   (task_state 更新)       │
              └────────────┬──────────────┘
                           │
                           ▼
              ┌───────────────────────────┐
              │  Global State 更新        │
              │  + Toast 通知             │
              │  + 区域刷新               │
              └───────────────────────────┘
```

---

## 3. 前端设计

### 3.1 目录结构

```
app/static/js/
├── state/
│   ├── index.js          # 状态导出入口
│   ├── signals.js        # 信号实现
│   └── stores/
│       ├── taskStore.js      # 任务状态
│       ├── galleryStore.js   # 画廊状态
│       └── uiStore.js        # UI 状态（模态框、侧边栏等）
├── services/
│   ├── websocket.js     # WebSocket 管理
│   ├── api.js           # 统一 API 调用
│   └── operations.js    # 操作服务（封装 CRUD）
├── components/
│   ├── Toast.js         # Toast 组件
│   ├── Modal.js         # 模态框组件
│   └── Progress.js      # 进度条组件
└── main.js              # 入口文件
```

### 3.2 状态管理设计

```javascript
// stores/taskStore.js
import { Signal, computed } from '../signals.js';

export const taskStore = {
    // 原始信号
    _currentTask: new Signal(null),
    _queue: new Signal([]),
    
    // 计算属性
    isRunning: computed(() => this._currentTask.value !== null),
    progress: computed(() => this._currentTask.value?.progress_percent ?? 0),
    currentOperation: computed(() => this._currentTask.value?.current_operation ?? ''),
    
    // 动作
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
    
    clear() {
        this._currentTask.value = null;
    },
    
    // 订阅
    subscribe(callback) {
        return this._currentTask.subscribe(callback);
    },
};

// 导出便捷访问
export const { isRunning, progress, currentOperation } = taskStore;
```

```javascript
// stores/galleryStore.js
import { Signal, computed } from '../signals.js';

export const galleryStore = {
    // 当前路径
    currentPath: new Signal(''),
    
    // 当前视图模式
    viewMode: new Signal('folder'),  // folder | list | waterfall
    
    // 排序
    sortBy: new Signal('modified_at'),
    sortOrder: new Signal('desc'),
    
    // 筛选
    filters: new Signal({
        filename: '',
        sizeMin: '',
        sizeMax: '',
        dateFrom: '',
        dateTo: '',
        tag: '',
    }),
    
    // 分页
    page: new Signal(1),
    hasNext: new Signal(false),
    
    // 数据缓存
    _cache: new Map(),
    
    // 方法
    setPath(path) {
        this.currentPath.value = path;
        this.page.value = 1;
        this._cache.clear();
    },
    
    setViewMode(mode) {
        this.viewMode.value = mode;
    },
    
    updateFilters(newFilters) {
        this.filters.value = { ...this.filters.value, ...newFilters };
        this.page.value = 1;
    },
    
    // 缓存管理
    getCachedData(path, page) {
        const key = `${path}:${page}`;
        return this._cache.get(key);
    },
    
    setCachedData(path, page, data) {
        const key = `${path}:${page}`;
        this._cache.set(key, data);
    },
    
    invalidateCache(path = '') {
        if (path) {
            // 只清除指定路径的缓存
            for (const key of this._cache.keys()) {
                if (key.startsWith(path)) {
                    this._cache.delete(key);
                }
            }
        } else {
            this._cache.clear();
        }
    },
};
```

### 3.3 WebSocket 服务

```javascript
// services/websocket.js
import { taskStore } from '../state/stores/taskStore.js';
import { galleryStore } from '../state/stores/galleryStore.js';
import { showToast } from '../components/Toast.js';

class WebSocketService {
    constructor() {
        this.ws = null;
        this.reconnectInterval = 3000;
        this.maxReconnectAttempts = 5;
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
                // 操作完成后自动刷新
                galleryStore.invalidateCache();
                showToast(payload.result_message || '操作完成', 'success');
                break;
                
            case 'task_error':
                taskStore.fail(payload.error);
                showToast(payload.error, 'error', 6000);
                break;
                
            case 'gallery_update':
                // 通知画廊刷新
                galleryStore.invalidateCache(payload.affected_path);
                break;
                
            case 'notification':
                showToast(payload.message, payload.level || 'info');
                break;
                
            default:
                console.log('Unknown message type:', type);
        }
    }
    
    send(type, payload) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type, payload }));
        }
    }
}

// 单例导出
export const wsService = new WebSocketService();
```

### 3.4 统一 API 服务

```javascript
// services/api.js
import { wsService } from './websocket.js';

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
        headers: {
            'Content-Type': 'application/json',
        },
    };
    
    const mergedOptions = {
        ...defaultOptions,
        ...options,
        headers: {
            ...defaultOptions.headers,
            ...options.headers,
        },
    };
    
    const response = await fetch(url, mergedOptions);
    
    // 处理非 JSON 响应
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
    // 通用请求方法
    get(url, options = {}) {
        return request(url, { ...options, method: 'GET' });
    },
    
    post(url, body, options = {}) {
        return request(url, { ...options, method: 'POST', body: JSON.stringify(body) });
    },
    
    put(url, body, options = {}) {
        return request(url, { ...options, method: 'PUT', body: JSON.stringify(body) });
    },
    
    delete(url, options = {}) {
        return request(url, { ...options, method: 'DELETE' });
    },
    
    // 封装常用操作
    operations: {
        async deleteImages(ids) {
            return api.post('/api/delete-images', { ids });
        },
        
        async moveImages(ids, targetPath) {
            return api.post('/api/move-images', { ids, target_path: targetPath });
        },
        
        async uploadFiles(files, path, options = {}) {
            const formData = new FormData();
            formData.append('path', path);
            if (options.onDuplicate) {
                formData.append('on_duplicate', options.onDuplicate);
            }
            files.forEach(file => formData.append('files', file));
            
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData,
            });
            return response.json();
        },
        
        async createFolder(path, name) {
            return api.post('/api/folders', { path, name });
        },
        
        async renameFolder(oldPath, newName) {
            return api.put(`/api/folders/${encodeURIComponent(oldPath)}`, { name: newName });
        },
        
        async deleteFolders(paths) {
            return api.post('/api/delete-folders', { paths });
        },
    },
};
```

### 3.5 操作服务（结合乐观更新）

```javascript
// services/operations.js
import { api } from './api.js';
import { galleryStore } from '../state/stores/galleryStore.js';
import { showToast } from '../components/Toast.js';

class OperationService {
    constructor() {
        this.pendingOperations = new Map();
    }
    
    // 生成操作 ID
    generateOpId() {
        return `op_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }
    
    // 记录操作
    track(opId, type, data) {
        this.pendingOperations.set(opId, {
            type,
            data,
            startedAt: Date.now(),
        });
    }
    
    // 移除操作记录
    untrack(opId) {
        this.pendingOperations.delete(opId);
    }
    
    // 统一的操作执行器
    async execute(operationType, apiCall, options = {}) {
        const opId = this.generateOpId();
        const { 
            onSuccess, 
            onError, 
            onProgress,
            rollback, 
            successMessage,
            errorMessage 
        } = options;
        
        this.track(opId, operationType, options);
        
        try {
            // 执行 API 调用
            const result = await apiCall();
            
            // 成功后处理
            if (onSuccess) {
                onSuccess(result);
            }
            
            // 显示成功消息
            if (successMessage !== false) {
                showToast(successMessage || this.getDefaultSuccessMessage(operationType, result), 'success');
            }
            
            // 标记完成
            this.untrack(opId);
触发画廊刷新（延迟            
            // ，等待 WS 通知）
            setTimeout(() => {
                galleryStore.invalidateCache();
            }, 500);
            
            return result;
            
        } catch (error) {
            // 失败后处理
            if (onError) {
                onError(error);
            }
            
            // 回滚（如有）
            if (rollback) {
                await rollback();
            }
            
            // 显示错误消息
            showToast(errorMessage || error.message || '操作失败', 'error', 6000);
            
            // 标记完成
            this.untrack(opId);
            
            throw error;
        }
    }
    
    getDefaultSuccessMessage(operationType, result) {
        const messages = {
            'delete-images': `已删除 ${result.deleted} 项`,
            'move-images': `已移动 ${result.moved} 项`,
            'upload': `已上传 ${result.uploaded} 项${result.skipped ? `，${result.skipped} 项跳过` : ''}`,
            'create-folder': '文件夹已创建',
            'rename-folder': '文件夹已重命名',
            'delete-folders': `已删除 ${result.deleted_folders} 个文件夹`,
        };
        return messages[operationType] || '操作完成';
    }
    
    // 便捷方法：删除图片
    async deleteImages(ids) {
        return this.execute(
            'delete-images',
            () => api.operations.deleteImages(ids),
            {
                successMessage: `已删除 ${ids.length} 项`,
            }
        );
    }
    
    // 便捷方法：移动图片
    async moveImages(ids, targetPath) {
        return this.execute(
            'move-images',
            () => api.operations.moveImages(ids, targetPath),
            {
                successMessage: `已移动 ${ids.length} 项到 ${targetPath || '根目录'}`,
            }
        );
    }
    
    // 便捷方法：上传文件
    async uploadFiles(files, path, options = {}) {
        return this.execute(
            'upload',
            () => api.operations.uploadFiles(files, path, options),
            {
                successMessage: false, // 由 WS 通知
            }
        );
    }
}

export const operationService = new OperationService();
```

### 3.6 Toast 组件

```javascript
// components/Toast.js
class ToastContainer {
    constructor() {
        this.container = null;
        this.queue = [];
        this.isShowing = false;
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
        const toast = this.createToast(message, type);
        this.container.appendChild(toast);
        
        if (duration > 0) {
            setTimeout(() => {
                this.removeToast(toast);
            }, duration);
        }
    }
    
    createToast(message, type) {
        const el = document.createElement('div');
        
        const config = {
            success: {
                classes: 'bg-green-100 text-green-800 border border-green-200',
                icon: '<svg class="w-4 h-4 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg>',
            },
            error: {
                classes: 'bg-red-100 text-red-800 border border-red-200',
                icon: '<svg class="w-4 h-4 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path></svg>',
            },
            warning: {
                classes: 'bg-amber-100 text-amber-800 border border-amber-200',
                icon: '<svg class="w-4 h-4 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path></svg>',
            },
            info: {
                classes: 'bg-slate-800 text-white',
                icon: '<svg class="w-4 h-4 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>',
            },
        };
        
        const conf = config[type] || config.info;
        
        el.className = `px-4 py-3 rounded-lg shadow-lg text-sm font-medium max-w-md flex items-center gap-2.5 ${conf.classes}`;
        el.innerHTML = `${conf.icon}<span>${message}</span>`;
        
        return el;
    }
    
    removeToast(toast) {
        toast.style.opacity = '0';
        toast.style.transition = 'opacity 0.2s';
        setTimeout(() => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        }, 200);
    }
}

const toastContainer = new ToastContainer();

export function showToast(message, type = 'info', duration = 4000) {
    toastContainer.show(message, type, duration);
}
```

---

## 4. 后端设计

### 4.1 WebSocket 端点

```python
# app/routers/websocket.py
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from typing import Set
import json
import asyncio

router = APIRouter()


class ConnectionManager:
    """WebSocket 连接管理器"""
    
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
            disconnected = set()
            for connection in self.active_connections[client_id]:
                try:
                    await connection.send_json(message)
                except Exception:
                    disconnected.add(connection)
            # 清理断开的连接
            for conn in disconnected:
                self.active_connections[client_id].discard(conn)
    
    async def broadcast(self, message: dict):
        """广播消息到所有客户端"""
        for connections in self.active_connections.values():
            for connection in connections:
                try:
                    await connection.send_json(message)
                except Exception:
                    pass
    
    async def broadcast_to_path(self, message: dict, path: str):
        """广播消息到访问特定路径的客户端"""
        # 可根据客户端所在的页面路径进行过滤
        await self.broadcast(message)


manager = ConnectionManager()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket 端点"""
    # 简单实现：使用固定客户端 ID
    # 生产环境应使用会话/认证信息生成唯一 ID
    client_id = f"client_{id(websocket)}"
    await manager.connect(websocket, client_id)
    
    try:
        # 发送连接成功消息
        await websocket.send_json({
            "type": "connected",
            "payload": {"client_id": client_id},
            "timestamp": datetime.now().isoformat(),
        })
        
        while True:
            # 保持连接，可以处理客户端发来的消息
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
    """处理客户端发来的消息"""
    msg_type = message.get("type")
    payload = message.get("payload", {})
    
    if msg_type == "ping":
        await manager.send_personal({
            "type": "pong",
            "payload": {},
            "timestamp": datetime.now().isoformat(),
        }, client_id)
    elif msg_type == "subscribe":
        # 客户端订阅特定频道
        channel = payload.get("channel")
        # 实现频道订阅逻辑
        pass
```

### 4.2 消息广播服务

```python
# app/services/message_broadcaster.py
from datetime import datetime
from typing import Any
from enum import Enum
import json
import uuid

class MessageType(str, Enum):
    TASK_START = "task_start"
    TASK_PROGRESS = "task_progress"
    TASK_COMPLETE = "task_complete"
    TASK_ERROR = "task_error"
    GALLERY_UPDATE = "gallery_update"
    NOTIFICATION = "notification"


class MessageBroadcaster:
    """消息广播服务"""
    
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
    
    async def broadcast_task_start(
        self, 
        task_type: str, 
        title: str, 
        total_items: int = 0
    ):
        """广播任务开始"""
        message = self._create_message(
            MessageType.TASK_START,
            {
                "task_type": task_type,
                "title": title,
                "total_items": total_items,
            }
        )
        await self._manager.broadcast(message)
    
    async def broadcast_task_progress(
        self,
        task_type: str,
        processed_items: int,
        total_items: int,
        current_operation: str = "",
    ):
        """广播任务进度"""
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
    
    async def broadcast_task_complete(
        self,
        task_type: str,
        result: dict[str, Any],
        result_message: str = "",
    ):
        """广播任务完成"""
        message = self._create_message(
            MessageType.TASK_COMPLETE,
            {
                "task_type": task_type,
                "result": result,
                "result_message": result_message,
            }
        )
        await self._manager.broadcast(message)
    
    async def broadcast_task_error(
        self,
        task_type: str,
        error: str,
    ):
        """广播任务错误"""
        message = self._create_message(
            MessageType.TASK_ERROR,
            {
                "task_type": task_type,
                "error": error,
            }
        )
        await self._manager.broadcast(message)
    
    async def broadcast_gallery_update(
        self,
        affected_path: str = "",
        action: str = "update",
    ):
        """广播画廊更新"""
        message = self._create_message(
            MessageType.GALLERY_UPDATE,
            {
                "affected_path": affected_path,
                "action": action,
            }
        )
        await self._manager.broadcast(message)
    
    async def broadcast_notification(
        self,
        message: str,
        level: str = "info",  # info, success, warning, error
    ):
        """广播通知"""
        message = self._create_message(
            MessageType.NOTIFICATION,
            {
                "message": message,
                "level": level,
            }
        )
        await self._manager.broadcast(message)


# 全局单例
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
    PARTIAL = "partial"  # 部分成功


class ApiResponse(BaseModel, Generic[T]):
    """统一 API 响应格式"""
    
    status: ResponseStatus = ResponseStatus.SUCCESS
    message: str = ""
    data: Optional[T] = None
    affected: list[int] = []  # 受影响的资源 ID
    errors: list[str] = []     # 错误信息列表
    
    @classmethod
    def success(
        cls,
        data: T = None,
        message: str = "",
        affected: list[int] = None,
    ):
        return cls(
            status=ResponseStatus.SUCCESS,
            message=message,
            data=data,
            affected=affected or [],
        )
    
    @classmethod
    def error(
        cls,
        message: str,
        errors: list[str] = None,
    ):
        return cls(
            status=ResponseStatus.ERROR,
            message=message,
            errors=errors or [],
        )
    
    @classmethod
    def partial(
        cls,
        message: str,
        data: T = None,
        affected: list[int] = None,
        errors: list[str] = None,
    ):
        return cls(
            status=ResponseStatus.PARTIAL,
            message=message,
            data=data,
            affected=affected or [],
            errors=errors or [],
        )
```

### 4.4 任务服务集成

```python
# app/services/task_service.py
from typing import Any, Callable, Awaitable
from dataclasses import dataclass
from app.services.message_broadcaster import broadcaster
from app.services.task_state import task_state

@dataclass
class TaskContext:
    """任务上下文，包含任务元信息和广播服务"""
    
    task_type: str
    title: str
    total_items: int = 0
    processed_items: int = 0
    
    async def broadcast_start(self):
        await broadcaster.broadcast_task_start(
            self.task_type,
            self.title,
            self.total_items,
        )
    
    async def broadcast_progress(self, processed: int, operation: str = ""):
        self.processed_items = processed
        await broadcaster.broadcast_task_progress(
            self.task_type,
            processed,
            self.total_items,
            operation,
        )
    
    async def broadcast_complete(self, result: dict[str, Any], message: str = ""):
        await broadcaster.broadcast_task_complete(
            self.task_type,
            result,
            message,
        )
    
    async def broadcast_error(self, error: str):
        await broadcaster.broadcast_task_error(self.task_type, error)


class TaskService:
    """任务执行服务，封装任务执行和通知逻辑"""
    
    def __init__(self):
        self._handlers: dict[str, Callable[[TaskContext, Any], Awaitable[dict]]] = {}
    
    def register(self, task_type: str):
        """装饰器：注册任务处理器"""
        def decorator(func: Callable[[TaskContext, Any], Awaitable[dict]]):
            self._handlers[task_type] = func
            return func
        return decorator
    
    async def execute(
        self,
        task_type: str,
        title: str,
        params: Any = None,
        total_items: int = 0,
    ) -> dict[str, Any]:
        """执行任务并广播进度"""
        # 检查是否有任务在进行
        if not task_state.start_task(task_type, total_items, title):
            return {"error": "有任务正在进行中"}
        
        context = TaskContext(task_type=task_type, title=title, total_items=total_items)
        
        try:
            # 广播任务开始
            await context.broadcast_start()
            
            # 执行任务
            handler = self._handlers.get(task_type)
            if not handler:
                raise ValueError(f"Unknown task type: {task_type}")
            
            result = await handler(context, params)
            
            # 生成完成消息
            result_message = self._format_result_message(task_type, result)
            
            # 广播任务完成
            await context.broadcast_complete(result, result_message)
            
            # 更新任务状态
            task_state.end_task(result)
            
            return result
            
        except Exception as e:
            error_msg = str(e)
            
            # 广播任务错误
            await context.broadcast_error(error_msg)
            
            # 更新任务状态
            task_state.fail_task(error_msg)
            
            return {"error": error_msg}
    
    def _format_result_message(self, task_type: str, result: dict[str, Any]) -> str:
        """格式化结果消息"""
        messages = {
            "scan": f"扫描完成，发现 {result.get('scanned', 0)} 个文件",
            "cleanup": f"清理完成，移除 {result.get('stale_removed', 0)} 条记录",
            "upload": f"已上传 {result.get('uploaded', 0)} 个文件",
            "delete": f"已删除 {result.get('deleted', 0)} 项",
            "move-images": f"已移动 {result.get('moved', 0)} 项",
            "move-folders": f"已移动 {result.get('moved', 0)} 个文件夹",
        }
        return messages.get(task_type, "操作完成")


task_service = TaskService()
```

### 4.5 改造后的路由示例

```python
# app/routers/images.py (改造后)
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import get_async_session
from app.schemas.response import ApiResponse
from app.services.task_service import task_service

router = APIRouter(prefix="/api", tags=["images"])


@task_service.register("delete-images")
async def handle_delete_images(
    context: TaskContext,
    params: dict,
) -> dict:
    """删除图片任务处理器"""
    session: AsyncSession = params["session"]
    image_ids: list[int] = params["ids"]
    
    # 模拟删除过程
    deleted_count = 0
    for i, image_id in enumerate(image_ids):
        # 删除逻辑...
        deleted_count += 1
        
        # 每处理 10 个广播一次进度
        if i % 10 == 0:
            await context.broadcast_progress(i + 1, f"正在删除第 {i + 1} 项...")
            await asyncio.sleep(0)  # 让出控制权
    
    return {"deleted": deleted_count}


@router.post("/delete-images")
async def delete_images(
    body: DeleteImagesRequest,
    session: AsyncSession = Depends(get_async_session),
):
    """删除图片 API"""
    result = await task_service.execute(
        task_type="delete-images",
        title="正在删除文件",
        params={"session": session, "ids": body.ids},
        total_items=len(body.ids),
    )
    
    if "error" in result:
        return ApiResponse.error(result["error"])
    
    return ApiResponse.success(
        data={"deleted": result.get("deleted", 0)},
        message=f"已删除 {result.get('deleted', 0)} 项",
        affected=body.ids,
    )
```

---

## 5. 前后端交互流程

### 5.1 典型操作流程：删除图片

```
1. 用户点击删除按钮
         │
         ▼
2. 前端显示确认对话框
         │
         ▼
3. 用户确认 → 调用 operationService.deleteImages(ids)
         │
         ▼
4. operationService 调用 api.deleteImages(ids)
         │
         ▼
5. 后端处理请求
         │
         ├── 5.1 调用 task_service.execute()
         │         │
         │         ▼
         │    广播 task_start 消息
         │         │
         │         ▼
         │    执行删除逻辑，定期广播 task_progress
         │         │
         │         ▼
         │    广播 task_complete 消息
         │         │
         │         ▼
         │    返回结果
         │
         ▼
6. 前端接收 HTTP 响应
         │
         ▼
7. 显示成功 Toast（可选，因为 WS 也会通知）
         │
         ▼
8. WebSocket 接收 task_complete 消息
         │
         ├── 8.1 更新 taskStore
         │         │
         │         ▼
         │    显示 Toast
         │         │
         │         ▼
         │    触发 galleryStore 缓存失效
         │         │
         │         ▼
         │    自动刷新画廊
         │
         ▼
9. 用户看到更新后的画廊
```

### 5.2 错误处理流程

```
1. 操作失败（API 返回错误）
         │
         ▼
2. operationService.catch 捕获错误
         │
         ├── 2.1 调用 onError 回调
         │
         ├── 2.2 执行 rollback（如有）
         │
         ├── 2.3 显示错误 Toast
         │
         ▼
3. 结束
```

### 5.3 页面导航后重连

```javascript
// main.js
import { wsService } from './services/websocket.js';

// HTMX 页面切换后重连 WebSocket
document.addEventListener('htmx:afterSwap', (event) => {
    // 检查是否是主要内容区更新
    if (event.detail.target.id === 'gallery-container' || 
        event.detail.target.id === 'main-content') {
        
        // 确保 WebSocket 连接正常
        if (!wsService.ws || wsService.ws.readyState !== WebSocket.OPEN) {
            wsService.connect();
        }
    }
});

// 页面卸载时断开连接
window.addEventListener('beforeunload', () => {
    wsService.disconnect();
});
```

---

## 6. 渐进式实施方案

考虑到完全重构的工作量，建议分阶段实施：

### 阶段 1：基础设施（1-2 周）

1. 实现 WebSocket 端点和连接管理
2. 实现消息广播服务
3. 实现前端 WebSocket 服务和信号系统
4. 实现 Toast 组件

### 阶段 2：任务通知迁移（1 周）

1. 将现有 SSE 替换为 WebSocket
2. 改造 task_service 执行服务
3. 前端任务状态同步

### 阶段 3：操作服务改造（1-2 周）

1. 实现统一 API 响应格式
2. 实现 operationService
3. 改造现有操作（删除、移动、上传）

### 阶段 4：UI 状态优化（1 周）

1. 实现 galleryStore
2. 实现缓存管理
3. 优化刷新逻辑

### 阶段 5：完善和优化（持续）

1. 添加重连机制
2. 优化性能
3. 添加单元测试

---

## 7. 风险和注意事项

### 7.1 兼容性

- WebSocket 在某些企业网络环境下可能被阻断，应保留 HTTP 轮询作为后备方案
- 考虑移动网络下的电量消耗

### 7.2 安全性

- WebSocket 连接需要身份验证
- 防止跨站 WebSocket 攻击

### 7.3 性能

- 大量并发连接时需要考虑性能优化
- 消息频率控制，避免刷屏

### 7.4 回退策略

- 新架构出问题时能快速回退到现有方案
- 保持 API 兼容

---

## 8. 文件清单

改造过程中需要新增/修改的文件：

### 新增文件

```
app/
├── routers/
│   └── websocket.py          # WebSocket 端点
├── services/
│   ├── message_broadcaster.py  # 消息广播服务
│   └── task_service.py         # 任务执行服务
└── schemas/
    └── response.py              # 统一响应格式

app/static/js/
├── state/
│   ├── index.js
│   ├── signals.js
│   └── stores/
│       ├── taskStore.js
│       ├── galleryStore.js
│       └── uiStore.js
├── services/
│   ├── websocket.js
│   ├── api.js
│   └── operations.js
├── components/
│   ├── Toast.js
│   ├── Modal.js
│   └── Progress.js
└── main.js

tests/
├── test_websocket.py
├── test_operations.py
└── test_stores.js
```

### 修改文件

```
app/main.py                      # 注册 WebSocket 路由
app/routers/images.py           # 使用新任务服务
app/routers/folders.py          # 使用新任务服务
app/routers/settings.py          # 移除 SSE，使用 WS
app/templates/base.html          # 引入新的 JS 模块
app/static/js/gallery.js        # 逐步迁移到新架构
```

---

## 9. 测试策略

### 9.1 单元测试

```python
# tests/test_operations.py
import pytest
from app.services.operations import OperationService

@pytest.mark.asyncio
async def test_delete_images_success():
    # 模拟 API 返回
    # 执行操作
    # 验证结果
    pass

@pytest.mark.asyncio
async def test_delete_images_rollback():
    # 模拟删除失败
    # 验证回滚逻辑
    pass
```

```javascript
// tests/test_stores.js
import { test } from 'vitest';
import { galleryStore } from '../app/static/js/state/stores/galleryStore.js';

test('galleryStore setPath clears page', () => {
    galleryStore.page.value = 5;
    galleryStore.setPath('test/path');
    expect(galleryStore.page.value).toBe(1);
});
```

### 9.2 集成测试

- WebSocket 连接测试
- 消息广播测试
- 端到端操作流程测试

---

## 10. 参考资料

- [FastAPI WebSocket](https://fastapi.tiangolo.com/advanced/websockets/)
- [Signals 模式](https://preactjs.com/guide/v10/signals/)
- [HTMX WebSocket](https://htmx.org/extensions/web-sockets/)
