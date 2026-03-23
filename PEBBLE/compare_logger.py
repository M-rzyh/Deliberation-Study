import os
from datetime import datetime

def _base_log_dir():
    """Resolve base log directory with priority for per-job paths."""
    compare_dir = os.environ.get("COMPARE_RUN_DIR", "").strip()
    if compare_dir:
        os.makedirs(compare_dir, exist_ok=True)
        return compare_dir

    # Fallback: keep local logs, but scope by SLURM job when available.
    job_id = os.environ.get("SLURM_JOB_ID", "").strip()
    if job_id:
        local_job_dir = os.path.join(os.getcwd(), "logs", job_id)
        os.makedirs(local_job_dir, exist_ok=True)
        return local_job_dir

    local_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(local_dir, exist_ok=True)
    return local_dir

def algo_dir(name):
    log_dir = os.path.join(_base_log_dir(), name)
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
    return Clock(_base_log_dir())

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
