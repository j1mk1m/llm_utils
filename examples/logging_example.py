#!/usr/bin/env python3
"""
Logging Configuration Examples for LLM Utils

This file demonstrates various ways to configure logging for the LLM utilities,
including different log levels, file outputs, and custom formatting.
"""

import logging
import os
from datetime import datetime
from llm_utils import setup_logging, create_llm_client, get_logger


def example_basic_logging():
    """Example of basic logging setup."""
    print("=== Basic Logging Example ===")
    
    # Set up basic logging to console
    setup_logging(level="INFO")
    
    # Create a client - it will use the configured logger
    client = create_llm_client(default_model="gpt-3.5-turbo")
    
    # This will log INFO level messages
    print("Client created with basic logging")
    print("Check console output for log messages\n")


def example_file_logging():
    """Example of logging to a file."""
    print("=== File Logging Example ===")
    
    # Set up logging to both console and file
    log_file = f"llm_utils_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    setup_logging(level="INFO", log_file=log_file)
    
    # Create a client
    client = create_llm_client(default_model="gpt-3.5-turbo")
    
    print(f"Logging to file: {log_file}")
    print("Check the log file for detailed output\n")


def example_debug_logging():
    """Example of debug logging with prompts and responses."""
    print("=== Debug Logging Example ===")
    
    # Set up debug logging to see prompts and responses
    setup_logging(level="DEBUG", log_file="debug_example.log")
    
    # Create a client
    client = create_llm_client(default_model="gpt-3.5-turbo")
    
    print("Debug logging enabled - prompts and responses will be logged")
    print("Check debug_example.log for detailed output\n")


def example_custom_logging():
    """Example of custom logging configuration."""
    print("=== Custom Logging Example ===")
    
    # Create a custom logger
    custom_logger = logging.getLogger("my_llm_app")
    custom_logger.setLevel(logging.INFO)
    
    # Create custom formatter
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    custom_logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler("custom_llm.log")
    file_handler.setFormatter(formatter)
    custom_logger.addHandler(file_handler)
    
    # Create client with custom logger
    client = create_llm_client(
        default_model="gpt-3.5-turbo",
        logger=custom_logger
    )
    
    print("Custom logging configuration applied")
    print("Check custom_llm.log for output\n")


def example_multiple_loggers():
    """Example of using multiple loggers for different components."""
    print("=== Multiple Loggers Example ===")
    
    # Set up main logger
    main_logger = setup_logging(level="INFO", log_file="main.log")
    
    # Create separate loggers for different components
    tracker_logger = get_logger("llm_utils.tracker")
    client_logger = get_logger("llm_utils.client")
    
    # Configure tracker logger to be more verbose
    tracker_logger.setLevel(logging.DEBUG)
    tracker_handler = logging.FileHandler("tracker.log")
    tracker_handler.setFormatter(logging.Formatter(
        "%(asctime)s [TRACKER] %(levelname)s: %(message)s"
    ))
    tracker_logger.addHandler(tracker_handler)
    
    # Configure client logger for errors only
    client_logger.setLevel(logging.ERROR)
    client_handler = logging.FileHandler("client_errors.log")
    client_handler.setFormatter(logging.Formatter(
        "%(asctime)s [CLIENT] ERROR: %(message)s"
    ))
    client_logger.addHandler(client_handler)
    
    # Create client with specific logger
    client = create_llm_client(
        default_model="gpt-3.5-turbo",
        logger=client_logger
    )
    
    print("Multiple loggers configured:")
    print("- Main logs: main.log")
    print("- Tracker logs: tracker.log")
    print("- Client errors: client_errors.log\n")


def example_logging_levels():
    """Example demonstrating different logging levels."""
    print("=== Logging Levels Example ===")
    
    # Test different logging levels
    levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
    
    for level in levels:
        print(f"\n--- Testing {level} level ---")
        
        # Set up logging for this level
        logger = setup_logging(level=level, log_file=f"level_{level.lower()}.log")
        
        # Create client
        client = create_llm_client(default_model="gpt-3.5-turbo", logger=logger)
        
        # Log messages at different levels
        logger.debug("This is a DEBUG message")
        logger.info("This is an INFO message")
        logger.warning("This is a WARNING message")
        logger.error("This is an ERROR message")
        
        print(f"Check level_{level.lower()}.log for {level} level output")


def example_structured_logging():
    """Example of structured logging with JSON format."""
    print("=== Structured Logging Example ===")
    
    import json
    
    class JSONFormatter(logging.Formatter):
        def format(self, record):
            log_entry = {
                "timestamp": self.formatTime(record),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno
            }
            return json.dumps(log_entry)
    
    # Set up structured logging
    logger = logging.getLogger("structured_llm")
    logger.setLevel(logging.INFO)
    
    # JSON formatter
    json_handler = logging.FileHandler("structured_llm.jsonl")
    json_handler.setFormatter(JSONFormatter())
    logger.addHandler(json_handler)
    
    # Create client
    client = create_llm_client(default_model="gpt-3.5-turbo", logger=logger)
    
    print("Structured logging enabled - check structured_llm.jsonl")
    print("Each log entry is a JSON object for easy parsing\n")


def cleanup_log_files():
    """Clean up example log files."""
    log_files = [
        "debug_example.log",
        "custom_llm.log", 
        "main.log",
        "tracker.log",
        "client_errors.log",
        "structured_llm.jsonl"
    ]
    
    # Add level-specific log files
    for level in ["debug", "info", "warning", "error"]:
        log_files.append(f"level_{level}.log")
    
    # Add timestamped log files
    for file in os.listdir("."):
        if file.startswith("llm_utils_") and file.endswith(".log"):
            log_files.append(file)
    
    print("Cleaning up example log files...")
    for log_file in log_files:
        if os.path.exists(log_file):
            os.remove(log_file)
            print(f"Removed {log_file}")
    print("Cleanup complete\n")


if __name__ == "__main__":
    print("LLM Utils Logging Examples")
    print("=" * 50)
    
    try:
        # Run examples
        example_basic_logging()
        example_file_logging()
        example_debug_logging()
        example_custom_logging()
        example_multiple_loggers()
        example_logging_levels()
        example_structured_logging()
        
        print("All examples completed!")
        print("\nTo clean up log files, run: cleanup_log_files()")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()
    
    # Uncomment to clean up log files
    # cleanup_log_files()
