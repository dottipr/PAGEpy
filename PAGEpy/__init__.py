import logging


class MyFormatter(logging.Formatter):
    info_format = "\033[32m%(asctime)s [%(module)s:%(lineno)d]\033[0m %(message)s"
    error_format = "\033[31m%(asctime)s [%(module)s:%(lineno)d] [%(levelname)s]\033[0m %(message)s"

    def format(self, record):
        if record.levelno > logging.INFO:
            fmt = self.error_format
        else:
            fmt = self.info_format
        formatter = logging.Formatter(fmt, datefmt="%H:%M:%S")
        return formatter.format(record)


handler = logging.StreamHandler()
handler.setFormatter(MyFormatter())
logging.basicConfig(level=logging.INFO, handlers=[handler])

# logger = logging.getLogger("PAGEpy")

# logger.info("This is an info message.")
# logger.warning("This is a warning.")
# logger.error("This is an error.")
