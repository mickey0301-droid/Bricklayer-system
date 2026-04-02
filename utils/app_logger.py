import logging
import os
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
_LOG_DIR = os.path.join(_ROOT, "logs")
_LOG_FILE = os.path.join(_LOG_DIR, "app.log")
_LOGGER_NAME = "bricklayer_app"


def _get_logger() -> logging.Logger:
    logger = logging.getLogger(_LOGGER_NAME)
    if logger.handlers:
        return logger

    os.makedirs(_LOG_DIR, exist_ok=True)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(_LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    # 仍保留 stdout，方便 Streamlit logs 看到
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    return logger


def log_info(message: str):
    _get_logger().info(message)


def log_error(message: str):
    _get_logger().error(message)


def log_exception(message: str):
    _get_logger().exception(message)


def get_log_path() -> str:
    return _LOG_FILE


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
