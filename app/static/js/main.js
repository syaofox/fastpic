import { wsService } from './services/websocket.js';
import { taskStore } from './state/stores/taskStore.js';
import { galleryStore } from './state/stores/galleryStore.js';
import { selectionStore } from './state/stores/selectionStore.js';
import { showToast } from './components/Toast.js';
import { progressIndicator } from './components/Progress.js';
import { operationService } from './services/operations.js';

document.addEventListener('DOMContentLoaded', () => {
    wsService.connect();
    
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

document.addEventListener('htmx:afterSwap', (event) => {
    if (event.detail.target.id === 'gallery-container' || 
        event.detail.target.id === 'main-content') {
        if (!wsService.ws || wsService.ws.readyState !== WebSocket.OPEN) {
            wsService.connect();
        }
    }
});

window.addEventListener('beforeunload', () => {
    wsService.disconnect();
});

window.galleryStore = galleryStore;
window.selectionStore = selectionStore;
window.showToast = showToast;
window.wsService = wsService;
window.operationService = operationService;
