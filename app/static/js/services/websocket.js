import { taskStore } from '../state/stores/taskStore.js';
import { galleryStore } from '../state/stores/galleryStore.js';

class WebSocketService {
    constructor() {
        this.ws = null;
        this.reconnectInterval = 3000;
        this.maxReconnectAttempts = 10;
        this.reconnectAttempts = 0;
        this.shouldReconnect = true;
        this.connecting = false;
    }
    
    connect() {
        if (this.connecting || (this.ws && this.ws.readyState !== WebSocket.CLOSED)) {
            return;
        }
        this.connecting = true;
        
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        
        this.ws = new WebSocket(wsUrl);
        
        this.ws.onopen = () => {
            console.log('WebSocket connected');
            this.reconnectAttempts = 0;
            this.connecting = false;
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
            this.connecting = false;
            if (this.shouldReconnect && this.reconnectAttempts < this.maxReconnectAttempts) {
                this.reconnectAttempts++;
                setTimeout(() => this.connect(), this.reconnectInterval);
            }
        };
        
        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.connecting = false;
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
            case 'connected':
                console.log('WS connected:', payload);
                break;
                
            case 'task_start':
            case 'task_progress':
            case 'task_complete':
            case 'task_error':
                this.handleTaskMessage(type, payload);
                break;
                
            case 'gallery_update':
                galleryStore.invalidateCache(payload.affected_path);
                break;
                
            case 'scan_status':
                this.handleScanStatus(payload);
                break;
                
            case 'notification':
                if (typeof window.showToast === 'function') {
                    window.showToast(payload.message, payload.level || 'info');
                }
                break;
                
            default:
                console.log('Unknown message type:', type);
        }
    }

    handleTaskMessage(type, payload) {
        const { task_id, task_type, title, total_items, processed_items, current_operation, result, result_message, error } = payload;
        
        // Build unified task object for the store from any message type
        const task = {
            id: task_id || `ws_${Date.now()}`,
            task_type: task_type || '',
            title: title || '',
            total_items: total_items || 0,
            completed_items: processed_items || 0,
            current_operation: current_operation || '',
        };

        // Copy over the existing task fields if we have one
        const existing = taskStore.tasks.find(t => t.id === task.id);
        if (existing) {
            Object.assign(task, existing, task);
        }

        switch (type) {
            case 'task_start':
                task.status = 'pending';
                task.progress_percent = 0;
                task.created_at = Date.now() / 1000;
                taskStore.addOrUpdateTask(task);
                break;

            case 'task_progress':
                task.status = 'running';
                task.progress_percent = total_items > 0
                    ? Math.round((processed_items / total_items) * 100)
                    : 0;
                task.completed_items = processed_items;
                task.total_items = total_items;
                task.current_operation = current_operation || '';
                taskStore.addOrUpdateTask(task);
                break;

            case 'task_complete':
                task.status = 'completed';
                task.progress_percent = 100;
                task.finished_at = Date.now() / 1000;
                task.result_summary = result_message || '';
                if (result) {
                    task.completed_items = result.completed_items || task.completed_items;
                }
                taskStore.addOrUpdateTask(task);
                galleryStore.invalidateCache();
                if (result_message && typeof window.showToast === 'function') {
                    window.showToast(result_message, 'success');
                }
                break;

            case 'task_error':
                task.status = 'failed';
                task.finished_at = Date.now() / 1000;
                task.error_message = error || '';
                task.current_operation = '任务失败';
                taskStore.addOrUpdateTask(task);
                if (typeof window.showToast === 'function') {
                    window.showToast(error || '任务失败', 'error', 6000);
                }
                break;
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
