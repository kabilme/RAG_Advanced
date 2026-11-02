"""Frontend launcher script."""

import subprocess
import sys

if __name__ == "__main__":
    app_path = "app/ui/streamlit_app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", app_path, "--server.port", "8501", "--server.headless", "true"]
    print(f"Launching Streamlit Frontend on http://localhost:8501 ...")
    subprocess.run(cmd)
