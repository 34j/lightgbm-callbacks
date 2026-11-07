import subprocess
import sys

if sys.platform == "darwin":
    subprocess.run(["brew", "install", "libomp"], check=True)
