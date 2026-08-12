#!/usr/bin/env python3
"""
Telemetry Listener – captures raw inbound JSON from Cline / Continue dev.

Usage:
    python telemetry.py

Then configure your extension:
    Provider: OpenAI
    API Base: http://127.0.0.1:8000/v1
    API Key:  anything
    Model:    deepseek-chat

Trigger a chat or apply command. The terminal will display every detail.
"""

import json
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

# ==================== Mock Request Handler ====================
class TelemetryHandler(BaseHTTPRequestHandler):
    """
    Handles all incoming HTTP requests, logs them, and returns a dummy 200.
    """

    def log_request_details(self, method):
        """Parse and pretty-print the request data."""
        print("\n" + "=" * 70)
        print(f"[{self.log_date_time_string()}] {method} {self.path}")
        print("-" * 70)

        # Headers
        print("HEADERS:")
        for key, value in self.headers.items():
            print(f"  {key}: {value}")

        # Body (if present)
        content_length = self.headers.get('Content-Length')
        if content_length:
            body = self.rfile.read(int(content_length))
            try:
                body_json = json.loads(body)
                print("\nBODY (JSON):")
                print(json.dumps(body_json, indent=2, ensure_ascii=False))
            except Exception:
                print("\nBODY (raw):")
                print(body.decode("utf-8", errors="replace"))
        else:
            print("\nBODY: (empty)")

        print("=" * 70 + "\n")

    def send_dummy_response(self):
        """Send a minimal 200 response to prevent extension errors."""
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"status": "ok"}).encode())

    def do_GET(self):
        """Handle GET requests (e.g., /v1/models)."""
        self.log_request_details("GET")
        # Return a fake models list so the extension doesn't complain
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        response = {
            "object": "list",
            "data": [
                {
                    "id": "deepseek-chat",
                    "object": "model",
                    "created": 1700000000,
                    "owned_by": "telemetry"
                }
            ]
        }
        self.wfile.write(json.dumps(response).encode())

    def do_POST(self):
        """Handle POST requests (e.g., /v1/chat/completions)."""
        self.log_request_details("POST")
        self.send_dummy_response()

    def do_OPTIONS(self):
        """Handle preflight CORS if needed."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    # Suppress default logging to stderr
    def log_message(self, format, *args):
        pass  # we print our own summary


# ==================== Main ====================
if __name__ == "__main__":
    PORT = 8000
    server = HTTPServer(("127.0.0.1", PORT), TelemetryHandler)
    print(f"Telemetry listener running on http://127.0.0.1:{PORT}")
    print("Configure your VS Code extension:")
    print("  API Base: http://127.0.0.1:8000/v1")
    print("  API Key:  any-string")
    print("  Model:    deepseek-chat")
    print("\nTrigger a request from Cline or Continue dev.")
    print("All request details will be printed below.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down telemetry listener.")
        server.server_close()