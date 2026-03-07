#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Parse arguments
COMMAND="${1:-start}"
DAEMON=""

shift || true

while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--daemon)
            DAEMON="yes"
            shift
            ;;
        -h|--help)
            COMMAND="help"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

show_help() {
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  start [opts]    Start the development environment (default)"
    echo "  stop            Stop MariaDB and FastPic"
    echo "  restart         Restart the environment"
    echo "  status          Check service status"
    echo "  build:css       Build CSS (Tailwind)"
    echo "  watch:css       Watch and build CSS in development"
    echo "  help            Show this help message"
    echo ""
    echo "Options:"
    echo "  -d, --daemon    Run uvicorn in background (daemon mode)"
    echo ""
    echo "Examples:"
    echo "  $0 start               # Start DB and run app in foreground"
    echo "  $0 start -d            # Start DB and run app in background"
    echo "  $0 stop                # Stop everything"
    echo "  $0 restart -d          # Restart in daemon mode"
    echo "  $0 build:css           # Build Tailwind CSS"
    echo "  $0 watch:css           # Watch mode for CSS development"
}

stop_services() {
    echo -e "${YELLOW}[1/2]${NC} Stopping FastPic..."
    pkill -f 'uvicorn app.main:app' 2>/dev/null || true
    fuser -k 8000/tcp 2>/dev/null || true
    echo -e "${GREEN}FastPic stopped${NC}"
    
    echo -e "${YELLOW}[2/2]${NC} Stopping MariaDB..."
    docker compose -f docker-compose.dev.yml down 2>/dev/null || true
    echo -e "${GREEN}MariaDB stopped${NC}"
}

build_css() {
    echo -e "${YELLOW}Building CSS...${NC}"
    npm run build:css
    echo -e "${GREEN}CSS built successfully!${NC}"
}

watch_css() {
    echo -e "${YELLOW}Watching CSS changes...${NC}"
    echo "Press Ctrl+C to stop"
    npm run watch:css
}

check_status() {
    local running=false
    
    if pgrep -f 'uvicorn app.main:app' >/dev/null 2>&1; then
        echo -e "${GREEN}FastPic: running${NC}"
        running=true
    else
        echo -e "${RED}FastPic: stopped${NC}"
    fi
    
    if docker ps --format '{{.Names}}' | grep -q "fastpic-mariadb-dev"; then
        echo -e "${GREEN}MariaDB: running${NC}"
        running=true
    else
        echo -e "${RED}MariaDB: stopped${NC}"
    fi
    
    if [[ "$running" == "true" ]]; then
        echo ""
        echo "Access at: http://localhost:8000"
    fi
}

start_services() {
    # Stop existing FastPic process
    echo -e "${YELLOW}[1/5]${NC} Stopping existing FastPic..."
    pkill -f 'uvicorn app.main:app' 2>/dev/null || true
    # Also kill any process using port 8000
    fuser -k 8000/tcp 2>/dev/null || true
    sleep 1

    # Stop existing containers
    echo -e "${YELLOW}[2/5]${NC} Stopping existing containers..."
    docker compose -f docker-compose.dev.yml down 2>/dev/null || true

    # Start MariaDB
    echo -e "${YELLOW}[3/5]${NC} Starting MariaDB..."
    docker compose -f docker-compose.dev.yml up -d

    # Fix permissions for data-dev directory (owned by docker after startup)
    if [[ -d "$SCRIPT_DIR/data-dev" ]]; then
        sudo chown -R "$(id -u):$(id -g)" "$SCRIPT_DIR/data-dev" 2>/dev/null || true
    fi

    # Wait for MariaDB to be ready
    echo -e "${YELLOW}[4/5]${NC} Waiting for MariaDB to be ready..."
    MAX_RETRIES=30
    RETRY_COUNT=0
    while [[ $RETRY_COUNT -lt $MAX_RETRIES ]]; do
        if docker exec fastpic-mariadb-dev mariadb-admin ping -h localhost -u root -pfastpic --silent 2>/dev/null; then
            echo -e "${GREEN}MariaDB is ready!${NC}"
            break
        fi
        RETRY_COUNT=$((RETRY_COUNT + 1))
        echo "  Waiting... ($RETRY_COUNT/$MAX_RETRIES)"
        sleep 1
    done

    if [[ $RETRY_COUNT -ge $MAX_RETRIES ]]; then
        echo -e "${RED}Failed to connect to MariaDB after $MAX_RETRIES seconds${NC}"
        exit 1
    fi

    # Set environment and start FastPic
    export MYSQL_HOST=127.0.0.1
    # Force polling to avoid inotify permission issues with docker data dirs
    export WATCHFILES_FORCE_POLLING=true

    echo -e "${YELLOW}[5/5]${NC} Starting FastPic..."

    if [[ "$DAEMON" == "yes" ]]; then
        # Daemon mode: use --reload for development
        nohup uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload > /tmp/fastpic.log 2>&1 &
        echo -e "${GREEN}FastPic started in background (PID: $(pgrep -f 'uvicorn app.main:app'))${NC}"
        echo "Logs: tail -f /tmp/fastpic.log"
    else
        # Foreground mode: no --reload to avoid watchfiles permission issues
        uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
    fi
}

# Handle commands
case "$COMMAND" in
    start)
        start_services
        ;;
    stop)
        stop_services
        ;;
    restart)
        stop_services
        sleep 1
        start_services
        ;;
    status)
        check_status
        ;;
    build:css)
        build_css
        ;;
    watch:css)
        watch_css
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo "Unknown command: $COMMAND"
        echo ""
        show_help
        exit 1
        ;;
esac
