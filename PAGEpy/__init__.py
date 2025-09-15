import logging
import os
import warnings
from typing import Optional

# 1. ENVIRONMENT SETUP (before TF import/use)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=INFO, 1=WARN, 2=ERROR, 3=FATAL
os.environ['TF_DETERMINISTIC_OPS'] = '1'   # Ensure reproducible results

# 2. SUPPRESS WARNINGS
warnings.filterwarnings('ignore', category=UserWarning, message='.*CUDA.*')


class ColoredFormatter(logging.Formatter):
    """Formatter with colors for terminal output"""

    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def format(self, record):
        # Choose format based on log level
        if record.levelno >= logging.ERROR:
            fmt = f"{self.COLORS.get(record.levelname, '')}%(asctime)s \
                [%(module)s:%(lineno)d] [%(levelname)s]{self.RESET} %(message)s"
        else:
            fmt = f"{self.COLORS.get(record.levelname, '')}%(asctime)s \
                [%(module)s:%(lineno)d]{self.RESET} %(message)s"

        formatter = logging.Formatter(fmt, datefmt="%H:%M:%S")
        return formatter.format(record)


class PlainFormatter(logging.Formatter):
    """Plain formatter without colors for file output"""

    def format(self, record):
        if record.levelno >= logging.ERROR:
            fmt = "%(asctime)s [%(module)s:%(lineno)d] [%(levelname)s] %(message)s"
        else:
            fmt = "%(asctime)s [%(module)s:%(lineno)d] %(message)s"

        formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    console_output: bool = True
) -> logging.Logger:
    """
    Set up logging configuration with optional file output and colored console output.

    Args:
        level: Logging level (default: INFO)
        log_file: Optional path to log file
        console_output: Whether to output to console (default: True)

    Returns:
        Configured logger instance
    """
    # Get the root logger and clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(level)

    handlers = []

    # Console handler with colors
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(ColoredFormatter())
        console_handler.setLevel(level)
        handlers.append(console_handler)

    # File handler without colors
    if log_file:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir:  # Only create directory if there's a path
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setFormatter(PlainFormatter())
        file_handler.setLevel(level)
        handlers.append(file_handler)

    # Add handlers to root logger
    for handler in handlers:
        root_logger.addHandler(handler)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module/script.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class JupyterHandler(logging.Handler):
    '''Create a custom handler that outputs to Jupyter cells.'''

    def emit(self, record):
        print(self.format(record))


def setup_jupyter_logging(level=logging.INFO):
    '''Setup logging specifically for Jupyter.'''
    # Clear existing handlers
    root = logging.getLogger()
    root.handlers.clear()

    # Create and configure the Jupyter handler
    jupyter_handler = JupyterHandler()

    # Simple formatter without colors (Jupyter handles styling)
    formatter = logging.Formatter(
        '%(asctime)s [%(name)s:%(lineno)d] [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    jupyter_handler.setFormatter(formatter)

    # Add handler to root logger
    root.addHandler(jupyter_handler)
    root.setLevel(level)

    return root


# Default setup - you can customize this
setup_logging(
    level=logging.INFO,
    console_output=True,
    log_file=None  # Set to a file path if you want file logging by default
)

# Suppress verbose library loggers
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("h5py").setLevel(logging.WARNING)

# logger = get_logger("PAGEpy")

# logger.debug("This is a debug message.")
# logger.info("This is an info message.")
# logger.warning("This is a warning.")
# logger.error("This is an error.")
# logger.critical("This is a critical message.")
