import time
import logging
from contextlib import contextmanager

@contextmanager
def track_time(operation_name: str, logger: logging.Logger):
    """
    A context manager to measure execution time with nanosecond precision.
    Logs the start and end of an operation automatically.
    """
    logger.info(f"⏳ {operation_name}...")
    start_ns = time.perf_counter_ns()
    
    try:
        yield
    finally:
        end_ns = time.perf_counter_ns()
        elapsed_ns = end_ns - start_ns
        elapsed_ms = elapsed_ns / 1_000_000
        logger.info(f"✅ {operation_name} completed in {elapsed_ms:.2f} ms ({elapsed_ns} ns)")