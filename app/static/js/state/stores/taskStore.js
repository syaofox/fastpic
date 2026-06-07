import { Signal, computed } from '../signals.js';

class TaskStore {
    constructor() {
        this._tasks = new Signal([]);
    }

    get tasks() { return this._tasks.value; }

    get activeTasks() {
        return this._tasks.value.filter(t => t.status === 'pending' || t.status === 'running');
    }

    get completedTasks() {
        return this._tasks.value.filter(t => t.status !== 'pending' && t.status !== 'running');
    }

    get activeCount() {
        return this.activeTasks.length;
    }

    setTasks(active, history) {
        const merged = [...(active || []), ...(history || [])];
        this._tasks.value = merged;
    }

    addOrUpdateTask(task) {
        const tasks = [...this._tasks.value];
        const idx = tasks.findIndex(t => t.id === task.id);
        if (idx >= 0) {
            tasks[idx] = { ...tasks[idx], ...task };
            this._tasks.value = tasks;
        } else {
            tasks.unshift(task);
            this._tasks.value = tasks;
        }
    }

    removeTask(taskId) {
        this._tasks.value = this._tasks.value.filter(t => t.id !== taskId);
    }

    clear() {
        this._tasks.value = [];
    }

    subscribe(callback) {
        return this._tasks.subscribe(callback);
    }
}

export const taskStore = new TaskStore();
