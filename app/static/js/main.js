import { wsService } from './services/websocket.js';
import { taskStore } from './state/stores/taskStore.js';
import { galleryStore } from './state/stores/galleryStore.js';
import { selectionStore } from './state/stores/selectionStore.js';
import { showToast } from './components/Toast.js';
import { progressIndicator } from './components/Progress.js';
import { operationService } from './services/operations.js';
import { api } from './services/api.js';

document.addEventListener('DOMContentLoaded', async () => {
    // 1. 连接 WebSocket
    wsService.connect();
    
    // 2. 获取服务器上的任务状态（页面刷新后恢复）
    try {
        const resp = await api.get('/api/tasks');
        if (resp && (resp.active || resp.history)) {
            taskStore.setTasks(resp.active || [], resp.history || []);
        }
    } catch (e) {
        // 静默失败，WebSocket 会接管
    }
    
    // 3. 监听从 WS 或初始加载来的任务，更新顶部进度条（仅活跃任务）
    taskStore.subscribe((tasks) => {
        const activeTasks = tasks.filter(t => t.status === 'pending' || t.status === 'running');
        // 顶部进度条只显示第一个活跃任务（兼容旧 UI)
        if (activeTasks.length > 0) {
            const t = activeTasks[0];
            if (!t.finished_at) {
                progressIndicator.show(
                    t.task_type,
                    t.title,
                    t.progress_percent || 0,
                    t.current_operation || ''
                );
            } else {
                progressIndicator.hide();
            }
        } else {
            progressIndicator.hide();
        }

        // 更新任务面板角标
        const badge = document.getElementById('task-badge');
        if (badge) {
            const count = activeTasks.length;
            if (count > 0) {
                badge.textContent = count > 99 ? '99+' : String(count);
                badge.classList.remove('hidden');
            } else {
                badge.classList.add('hidden');
            }
        }
    });
});

document.addEventListener('htmx:afterSwap', (event) => {
    if (event.detail.target.id === 'gallery-container' || 
        event.detail.target.id === 'main-content') {
        if (!wsService.ws || (wsService.ws.readyState !== WebSocket.OPEN && wsService.ws.readyState !== WebSocket.CONNECTING)) {
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
window.taskStore = taskStore;
// 触发面板订阅（如果 DOMContentLoaded 时 store 尚未就绪，面板订阅会错过）
if (typeof window._onTaskStoreReady === 'function') {
    window._onTaskStoreReady();
}
