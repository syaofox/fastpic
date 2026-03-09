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
                if (payload.result_message && typeof window.showToast === 'function') {
                    window.showToast(payload.result_message, 'success');
                }
                break;
                
            case 'task_error':
                taskStore.fail(payload.error);
                if (typeof window.showToast === 'function') {
                    window.showToast(payload.error, 'error', 6000);
                }
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
