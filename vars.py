import os

REQUEST_TIMEOUT = 600  # seconds
DEFAULT_ENDPOINT = "http://127.0.0.1:8080"
DEFAULT_DATA_DIR = "./data"


def init_vars() -> dict:
    """Read environment variables and return a dictionary of variables."""

    vars = {
        "REQUEST_TIMEOUT": int(os.getenv("REQUEST_TIMEOUT", REQUEST_TIMEOUT)),
        "DEFAULT_ENDPOINT": os.getenv("DEFAULT_ENDPOINT", DEFAULT_ENDPOINT),
        "DEFAULT_DATA_DIR": os.getenv("DEFAULT_DATA_DIR", DEFAULT_DATA_DIR),
    }

    return vars
