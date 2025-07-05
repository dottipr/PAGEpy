def init_cuda():
    import tensorflow as tf
    
    # Set memory growth BEFORE any GPU operations
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # This must be done before any operations that use GPU
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Memory growth enabled for {len(gpus)} GPU(s)")
        except RuntimeError:
            # GPU already initialized, memory growth setting ignored
            print("GPU already initialized")
        
        print(f"GPU devices available: {len(gpus)}")
        return True
    else:
        print("No GPU devices found, using CPU")
        return False
