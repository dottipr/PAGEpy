import logging
import os
import warnings

# 1. ENVIRONMENT SETUP (before TF import/use)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=INFO, 1=WARN, 2=ERROR, 3=FATAL
os.environ['TF_DETERMINISTIC_OPS'] = '1'   # Ensure reproducible results

# 2. SUPPRESS WARNINGS
# warnings.filterwarnings('ignore')  # General warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*CUDA.*')
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("h5py").setLevel(logging.WARNING)


class MyFormatter(logging.Formatter):
    debug_format = "\033[36m%(asctime)s [%(pathname)s:%(lineno)d] [DEBUG]\033[0m %(message)s"
    info_format = "\033[32m%(asctime)s [%(module)s:%(lineno)d]\033[0m %(message)s"
    error_format = "\033[31m%(asctime)s [%(module)s:%(lineno)d] [%(levelname)s]\033[0m %(message)s"

    def format(self, record):
        if record.levelno == logging.DEBUG:
            fmt = self.debug_format
        elif record.levelno > logging.INFO:
            fmt = self.error_format
        else:
            fmt = self.info_format
        formatter = logging.Formatter(fmt, datefmt="%H:%M:%S")
        return formatter.format(record)


handler = logging.StreamHandler()
handler.setFormatter(MyFormatter())
# <-- change this to show/hide debug logs
logging.basicConfig(level=logging.INFO, handlers=[handler])


# logger = logging.getLogger("PAGEpy")

# logger.debug("This is a debug message.")
# logger.info("This is an info message.")
# logger.warning("This is a warning.")
# logger.error("This is an error.")
