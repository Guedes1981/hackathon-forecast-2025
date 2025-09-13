import os
from pathlib import Path

def resolve_project_dir():
    colab_default = Path("/content/drive/MyDrive/hackathon-forecast-2025")
    env = os.getenv("PROJECT_DIR")
    if env:
        return Path(env)
    if colab_default.exists():
        return colab_default
    return Path(".").resolve()
