# src/utils/logging_utils.py
import logging
import sys
import os

def setup_logging(log_level=logging.INFO, log_file=None):
    """
    Sets up basic Python logging configuration.

    Args:
        log_level: The logging level (e.g., logging.INFO, logging.DEBUG).
        log_file: Optional path to a file to save logs.
    """
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )
    logging.info("Logging setup complete.")

if __name__ == '__main__':
    # Example usage:
    log_path = os.path.join("../..", "..", "results", "test_run", "training.log") # Example path
    setup_logging(log_level=logging.DEBUG, log_file=log_path)
    logging.debug("This is a debug message.")
    logging.info("This is an info message.")
    logging.warning("This is a warning message.")
    logging.error("This is an error message.")
    print(f"Check log file at: {os.path.abspath(log_path)}")

