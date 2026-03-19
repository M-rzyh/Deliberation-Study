import os
from datetime import datetime

def algo_dir(name):
    log_dir = os.path.join(os.getcwd(), "logs", name)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

class Clock:
    def __init__(self, log_dir):
        self.log_dir = log_dir
    def log_scalar(self, name, value, step):
        pass
    def flush(self):
        pass

def get_clock():
    return Clock(os.path.join(os.getcwd(), "logs"))

class Timer:
    def __init__(self):
        self.dt = 0
    def __enter__(self):
        import time
        self.start = time.time()
        return self
    def __exit__(self, *args):
        import time
        self.dt = time.time() - self.start
