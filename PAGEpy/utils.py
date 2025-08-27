import os
import platform

import tensorflow as tf
from tensorflow.keras import mixed_precision

from PAGEpy import get_logger

logger = get_logger(__name__)


def init_tensorflow():
    """
    Comprehensive TensorFlow initialization with GPU memory management
    and performance optimizations for both CUDA (Linux/Windows) and Metal (macOS).
    """
    # Detect platform
    is_macos = platform.system() == 'Darwin'
    is_apple_silicon = is_macos and platform.machine() == 'arm64'

    # GPU CONFIGURATION
    gpus = tf.config.list_physical_devices('GPU')
    gpu_available = False

    if gpus:
        logger.info("GPU devices available: %d", len(gpus))
        try:
            # Platform-specific GPU configuration
            if not is_macos:
                # CUDA-specific settings (Linux/Windows)
                os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
                logger.info("CUDA memory allocator configured")

            # Enable memory growth for all GPUs (works for both CUDA and Metal)
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("Memory growth enabled for %d GPU(s)", len(gpus))
            gpu_available = True

            # Mixed precision configuration
            if is_apple_silicon:
                # On Apple Silicon, use mixed_float16 cautiously
                # Some operations might not be supported
                try:
                    mixed_precision.set_global_policy('mixed_float16')
                    logger.info(
                        "Mixed precision (float16) enabled for Apple Silicon")
                except Exception as e:
                    logger.warning(
                        "Mixed precision failed on Apple Silicon: %s", e)
                    logger.info("Continuing without mixed precision")
            else:
                # CUDA generally handles mixed precision well
                mixed_precision.set_global_policy('mixed_float16')
                logger.info("Mixed precision (float16) enabled")

        except RuntimeError as e:
            logger.warning("GPU configuration failed: %s", e)
            logger.info("Falling back to CPU")
            gpu_available = False
    else:
        logger.info("No GPU devices found, using CPU")

    # CPU OPTIMIZATION (if no GPU or as fallback)
    if not gpu_available:
        # Optimize CPU threading
        tf.config.threading.set_inter_op_parallelism_threads(
            0)  # Use all cores
        tf.config.threading.set_intra_op_parallelism_threads(
            0)  # Use all cores
        logger.info("CPU threading optimized")

    # XLA CONFIGURATION - Platform specific
    if is_apple_silicon:
        # Disable XLA on Apple Silicon as it can cause platform registration errors
        logger.info("XLA disabled on Apple Silicon to avoid platform issues")
        # Optionally set environment variable to be extra sure
        os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'
    else:
        # Enable XLA on other platforms where it's more stable
        try:
            tf.config.optimizer.set_jit(True)
            logger.info("XLA compilation enabled")
        except Exception as e:
            logger.warning("XLA configuration failed: %s", e)

    return gpu_available
