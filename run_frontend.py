"""Frontend launcher script - launches lightweight responsive web application server."""

from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import sys

PORT = 8501
STATIC_DIR = Path(__file__).resolve().parent / "app" / "static"


class FrontendHandler(SimpleHTTPRequestHandler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def log_message(self, format, *args):
        # Quiet standard HTTP request logs for clean terminal output
        pass


if __name__ == "__main__":
    if not STATIC_DIR.exists():
        print(f"Error: Static directory {STATIC_DIR} not found.")
        sys.exit(1)

    print("=" * 65)
    print(f"🚀 Local Advanced RAG Responsive Web UI running on http://localhost:{PORT}")
    print(f"📡 Connected to Backend API on http://127.0.0.1:8000")
    print("=" * 65)
    httpd = HTTPServer(("0.0.0.0", PORT), FrontendHandler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down frontend server...")
