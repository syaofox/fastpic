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
