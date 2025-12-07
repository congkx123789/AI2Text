#!/bin/bash

# Demo script - Start frontend server để demo
# Đây là cách đơn giản nhất để xem frontend hoạt động

echo "🚀 Starting Frontend Demo Server..."
echo ""

cd "$(dirname "$0")"

# Start frontend server
python3 -m http.server 8080 &
FRONTEND_PID=$!

echo "✓ Frontend server started (PID: $FRONTEND_PID)"
echo ""
echo "🌐 Frontend is available at:"
echo "   http://localhost:8080/frontend.html"
echo ""
echo "📝 Note:"
echo "   - Frontend sẽ hiển thị UI để chọn model"
echo "   - Bạn cần start các API services riêng:"
echo "     * ai-llm-ss: Port 8001"
echo "     * ai-llm: Port 8000"
echo "     * AI2Text: Port 8002"
echo ""
echo "💡 Để start tất cả services (sau khi cài dependencies):"
echo "   python3 start_all_services.py start"
echo ""
echo "Press Ctrl+C to stop frontend server"

# Save PID
echo $FRONTEND_PID > /tmp/frontend_demo.pid

# Wait for interrupt
trap "echo ''; echo 'Stopping frontend server...'; kill $FRONTEND_PID 2>/dev/null; rm -f /tmp/frontend_demo.pid; exit" INT TERM

# Open browser
if command -v xdg-open > /dev/null; then
    sleep 2
    xdg-open http://localhost:8080/frontend.html 2>/dev/null &
elif command -v open > /dev/null; then
    sleep 2
    open http://localhost:8080/frontend.html 2>/dev/null &
fi

# Keep running
wait $FRONTEND_PID

