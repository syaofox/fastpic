import { Signal } from '../signals.js';

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
