"""Frontend launcher script - launches lightweight responsive web application server."""

from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import sys

PORT = 8501
STATIC_DIR = Path(__file__).resolve().parent / "app" / "static"


"""Frontend launcher script - launches lightweight responsive web application server."""

from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import sys

PORT = 8501
STATIC_DIR = Path(__file__).resolve().parent / "app" / "static"


class FrontendHandler(SimpleHTTPRequestHandler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def translate_path(self, path):
        """Normalize URL paths so that /static/... cleanly maps to STATIC_DIR/..."""
        url_path = path.split("?", 1)[0].split("#", 1)[0]
        if url_path in ("/widget/demo", "/demo", "/demo.html"):
            return str(STATIC_DIR / "demo.html")
        elif url_path in ("/widget/embed", "/embed", "/embed.html"):
            return str(STATIC_DIR / "embed.html")
        elif url_path in ("/widget/chat-widget.js", "/chat-widget.js", "/static/chat-widget.js"):
            return str(STATIC_DIR / "chat-widget.js")
        elif url_path.startswith("/static/"):
            url_path = url_path[len("/static"):]
        elif url_path == "/static":
            url_path = "/"
        return super().translate_path(url_path)

    def end_headers(self):
        # Prevent browser from aggressively caching 404s or stale assets
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        # Enable CORS for local cross-origin API calls
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, PUT, DELETE")
        self.send_header("Access-Control-Allow-Headers", "*")
        super().end_headers()

    def log_message(self, format, *args):
        # Clean terminal output for server requests
        sys.stderr.write(f"[{self.log_date_time_string()}] {self.address_string()} - {format % args}\n")



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
