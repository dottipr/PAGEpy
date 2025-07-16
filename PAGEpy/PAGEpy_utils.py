import logging


def init_cuda():
    import tensorflow as tf

    logger = logging.getLogger(__name__)

    # Set memory growth BEFORE any GPU operations
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("Memory growth enabled for %d GPU(s)", len(gpus))
        except RuntimeError:
            logger.warning(
                "GPU already initialized, memory growth setting ignored")

        logger.info("GPU devices available: %d", len(gpus))
        return True
    else:
        logger.info("No GPU devices found, using CPU")
        return False
