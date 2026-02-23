import os
from typing import Any, Dict

REQUEST_TIMEOUT = 600  # seconds
DEFAULT_ENDPOINT = "http://127.0.0.1:8080"
DEFAULT_DATA_DIR = "/mnt/gpfs/llm-datasets"


def init_vars() -> Dict[str, Any]:
    """Read environment variables and return a dictionary of variables."""

    vars = {
        "REQUEST_TIMEOUT": int(os.getenv("REQUEST_TIMEOUT", REQUEST_TIMEOUT)),
        "DEFAULT_ENDPOINT": os.getenv("DEFAULT_ENDPOINT", DEFAULT_ENDPOINT),
        "DEFAULT_DATA_DIR": os.getenv("DEFAULT_DATA_DIR", DEFAULT_DATA_DIR),
    }

    return vars
