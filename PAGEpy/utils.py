import os
import warnings
import logging
import tensorflow as tf
from tensorflow.keras import mixed_precision

# TODO: rename as e.g. init_tensorflow (from vs code)
def init_cuda():
    """
    Comprehensive TensorFlow initialization with GPU memory management
    and performance optimizations.
    """
    logger = logging.getLogger(__name__)
    
    # GPU CONFIGURATION (Set memory growth BEFORE any GPU operations)
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'  # Better memory management
    gpus = tf.config.list_physical_devices('GPU')
    gpu_available = False
    
    if gpus:
        logger.info("GPU devices available: %d", len(gpus))
        try:
            # Enable memory growth for all GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("Memory growth enabled for %d GPU(s)", len(gpus))
            gpu_available = True
            
            # 4. MIXED PRECISION (Only if GPU available)
            # This uses 16-bit floats for speed and memory efficiency
            mixed_precision.set_global_policy('mixed_float16')
            logger.info("Mixed precision (float16) enabled")
            
        except RuntimeError as e:
            logger.warning("GPU configuration failed: %s", e)
            logger.info("Falling back to CPU")
    else:
        logger.info("No GPU devices found, using CPU")
    
    # 5. CPU OPTIMIZATION (if no GPU or as fallback)
    if not gpu_available:
        # Optimize CPU threading
        tf.config.threading.set_inter_op_parallelism_threads(0)  # Use all cores
        tf.config.threading.set_intra_op_parallelism_threads(0)  # Use all cores
        logger.info("CPU threading optimized")
    
    # 6. ADDITIONAL PERFORMANCE SETTINGS
    # Enable XLA (Accelerated Linear Algebra) compilation for better performance
    tf.config.optimizer.set_jit(True)
    
    return gpu_available