class ToastContainer {
    constructor() {
        this.container = null;
        this.init();
    }
    
    init() {
        const existing = document.getElementById('toast-container');
        if (existing) {
            this.container = existing;
            return;
        }
        this.container = document.createElement('div');
        this.container.id = 'toast-container';
        this.container.className = 'fixed bottom-4 left-1/2 -translate-x-1/2 z-[120] flex flex-col gap-2 pointer-events-none';
        this.container.setAttribute('aria-live', 'polite');
        document.body.appendChild(this.container);
    }
    
    show(message, type = 'info', duration = 4000) {
        if (!this.container) {
            this.init();
        }
        if (!this.container) return;

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
