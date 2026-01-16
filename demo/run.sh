#!/bin/bash
# Demo runner - starts/stops both engine and demo servers

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PID_FILE="$SCRIPT_DIR/.demo.pids"

ENGINE_PORT=8001
DEMO_PORT=8002

start() {
    if [ -f "$PID_FILE" ]; then
        echo "Servers may already be running. Run '$0 stop' first."
        exit 1
    fi

    echo "Starting Polybos Media Engine on port $ENGINE_PORT..."
    cd "$PROJECT_DIR"
    python3.12 -m uvicorn polybos_engine.main:app --port $ENGINE_PORT &
    ENGINE_PID=$!

    echo "Starting Demo Server on port $DEMO_PORT..."
    python3.12 "$SCRIPT_DIR/server.py" &
    DEMO_PID=$!

    # Save PIDs
    echo "$ENGINE_PID" > "$PID_FILE"
    echo "$DEMO_PID" >> "$PID_FILE"

    echo ""
    echo "Servers started:"
    echo "  Engine: http://localhost:$ENGINE_PORT (PID: $ENGINE_PID)"
    echo "  Demo:   http://localhost:$DEMO_PORT (PID: $DEMO_PID)"
    echo ""
    echo "Open http://localhost:$DEMO_PORT in your browser"
    echo "Run '$0 stop' to stop both servers"

    # Wait for both processes
    wait
}

stop() {
    if [ ! -f "$PID_FILE" ]; then
        echo "No PID file found. Servers may not be running."
        # Try to kill by port anyway
        echo "Attempting to kill processes on ports $ENGINE_PORT and $DEMO_PORT..."
        lsof -ti:$ENGINE_PORT | xargs kill -9 2>/dev/null
        lsof -ti:$DEMO_PORT | xargs kill -9 2>/dev/null
        exit 0
    fi

    echo "Stopping servers..."
    while read -r pid; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "  Killing PID $pid"
            kill "$pid" 2>/dev/null
        fi
    done < "$PID_FILE"

    rm -f "$PID_FILE"
    echo "Servers stopped."
}

status() {
    echo "Checking server status..."

    if lsof -i:$ENGINE_PORT >/dev/null 2>&1; then
        echo "  Engine (port $ENGINE_PORT): RUNNING"
    else
        echo "  Engine (port $ENGINE_PORT): stopped"
    fi

    if lsof -i:$DEMO_PORT >/dev/null 2>&1; then
        echo "  Demo (port $DEMO_PORT): RUNNING"
    else
        echo "  Demo (port $DEMO_PORT): stopped"
    fi
}

case "$1" in
    start)
        start
        ;;
    stop)
        stop
        ;;
    restart)
        stop
        sleep 1
        start
        ;;
    status)
        status
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status}"
        echo ""
        echo "Commands:"
        echo "  start   - Start both engine and demo servers"
        echo "  stop    - Stop both servers"
        echo "  restart - Stop then start both servers"
        echo "  status  - Check if servers are running"
        exit 1
        ;;
esac
