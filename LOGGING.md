# Logging Guide for LLM Utils

This guide explains how to use the comprehensive logging features in the LLM Utils package.

## Overview

The LLM Utils package includes extensive logging capabilities to help you:
- Monitor API calls and responses
- Track usage and costs
- Debug issues with prompts and responses
- Monitor performance and errors
- Generate structured logs for analysis

## Quick Start

### Basic Logging Setup

```python
from llm_utils.llm_util import setup_logging, create_llm_client

# Set up basic logging
setup_logging(level="INFO")

# Create a client - it will automatically use logging
client = create_llm_client(default_model="gpt-3.5-turbo")

# Make API calls - they will be logged
response = client.chat_completion([
    {"role": "user", "content": "Hello, world!"}
])
```

### Debug Logging (Prompts and Responses)

```python
# Enable debug logging to see prompts and responses
setup_logging(level="DEBUG", log_file="debug.log")

client = create_llm_client(default_model="gpt-3.5-turbo")
response = client.chat_completion([
    {"role": "user", "content": "Tell me a joke"}
])
```

This will log:
- Request details (model, parameters)
- Full prompts and messages
- Response content (truncated for readability)
- Usage statistics (tokens, cost, response time)

## Logging Levels

### INFO Level
- API call initiation and completion
- Usage tracking (tokens, cost, response time)
- File operations (save/load usage data)
- Retry attempts

### DEBUG Level
- Full request messages and parameters
- Response content (first 200 characters)
- Detailed usage tracking
- File loading operations

### WARNING Level
- API errors that trigger retries
- File loading issues
- Rate limit warnings

### ERROR Level
- Failed API calls after all retries
- File save/load errors
- Unexpected exceptions

## Advanced Configuration

### Custom Logging Configuration

For more advanced logging needs, you can create custom loggers and configure them manually:

```python
import logging
from llm_utils.llm_util import create_llm_client, get_logger

# Create custom loggers
api_logger = get_logger("my_app.api")
tracker_logger = get_logger("my_app.tracker")

# Configure custom loggers
api_logger.setLevel(logging.INFO)
tracker_logger.setLevel(logging.DEBUG)

# Add custom handlers
file_handler = logging.FileHandler("custom_app.log")
formatter = logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
file_handler.setFormatter(formatter)
api_logger.addHandler(file_handler)

# Create client with custom logger
client = create_llm_client(default_model="gpt-3.5-turbo", logger=api_logger)
```

## Structured Logging

For structured logging (JSON format), you can create a custom formatter:

```python
import json
import logging
from llm_utils.llm_util import setup_logging, get_logger

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
logger = get_logger("structured_llm")
logger.setLevel(logging.INFO)

json_handler = logging.FileHandler("structured_llm.jsonl")
json_handler.setFormatter(JSONFormatter())
logger.addHandler(json_handler)

# Logs will be in JSON format:
# {"timestamp": "2024-01-01 12:00:00", "level": "INFO", "logger": "llm_utils.client", "message": "Making LLM request to gpt-3.5-turbo", "module": "llm_util", "function": "_make_request", "line": 490}
```

## Log Files

The logging system creates several types of log files:

### Main Log Files
- `llm_utils.log` - General application logs
- `llm_utils_tracker.log` - Usage tracking logs
- `llm_utils_client.log` - API client logs
- `llm_utils_errors.log` - Error logs only
- `llm_utils_debug.log` - Debug logs only

### Structured Logs
- `structured.jsonl` - JSON-formatted logs for analysis

### Performance Logs
- `performance.log` - Slow operation logs

## Log Rotation

Log files are automatically rotated when they reach 10MB, keeping 5 backup files:

```
llm_utils.log
llm_utils.log.1
llm_utils.log.2
llm_utils.log.3
llm_utils.log.4
llm_utils.log.5
```

## Examples

### Example 1: Basic Usage with Logging

```python
from llm_utils.llm_util import setup_logging, create_llm_client

# Set up logging
setup_logging(level="INFO", log_file="my_app.log")

# Create client
client = create_llm_client(
    default_model="gpt-3.5-turbo",
    default_temperature=0.7
)

# Make API calls
try:
    response = client.chat_completion([
        {"role": "user", "content": "What is the capital of France?"}
    ])
    print(response['choices'][0]['message']['content'])
except Exception as e:
    print(f"Error: {e}")

# Check usage statistics
stats = client.get_usage_stats()
print(f"Total requests: {stats.total_requests}")
print(f"Total cost: ${stats.total_cost:.4f}")
```

### Example 2: Debug Mode for Development

```python
from llm_utils.llm_util import setup_logging, create_llm_client

# Enable debug logging
setup_logging(level="DEBUG", log_file="debug.log")

client = create_llm_client(default_model="gpt-4")

# This will log full prompts and responses
response = client.chat_completion([
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain quantum computing in simple terms."}
])
```

### Example 3: Production Setup

```python
from llm_utils.llm_util import setup_logging, create_llm_client
import logging

# Set up production logging
setup_logging(
    level="INFO",
    log_file="/var/log/llm_utils/app.log"
)

# Create client
client = create_llm_client(default_model="gpt-3.5-turbo")

# Your application code here...
```

### Example 4: Custom Logger for Different Components

```python
from llm_utils.llm_util import create_llm_client, get_logger
import logging

# Create custom loggers
api_logger = get_logger("my_app.api")
tracker_logger = get_logger("my_app.tracker")

# Configure custom loggers
api_logger.setLevel(logging.INFO)
tracker_logger.setLevel(logging.DEBUG)

# Create client with custom logger
client = create_llm_client(
    default_model="gpt-3.5-turbo",
    logger=api_logger
)

# The tracker will use its own logger
client.usage_tracker.logger = tracker_logger
```

## Log Analysis

### Parsing Structured Logs

```python
import json
from datetime import datetime

def analyze_logs(log_file):
    """Analyze structured log file."""
    with open(log_file, 'r') as f:
        logs = [json.loads(line) for line in f]
    
    # Filter by level
    errors = [log for log in logs if log['level'] == 'ERROR']
    warnings = [log for log in logs if log['level'] == 'WARNING']
    
    # Filter by time range
    today = datetime.now().date()
    today_logs = [
        log for log in logs 
        if datetime.fromisoformat(log['timestamp']).date() == today
    ]
    
    # Analyze costs
    total_cost = sum(log.get('cost', 0) for log in logs if 'cost' in log)
    
    print(f"Total logs: {len(logs)}")
    print(f"Errors: {len(errors)}")
    print(f"Warnings: {len(warnings)}")
    print(f"Today's logs: {len(today_logs)}")
    print(f"Total cost: ${total_cost:.4f}")

# Usage
analyze_logs("structured_logs.jsonl")
```

### Monitoring with External Tools

The structured JSON logs can be easily integrated with log analysis tools:

- **ELK Stack**: Import JSON logs into Elasticsearch
- **Grafana**: Create dashboards for cost and usage monitoring
- **Splunk**: Parse structured logs for analysis
- **CloudWatch**: Send logs to AWS CloudWatch Logs

## Best Practices

### 1. Choose Appropriate Log Levels
- Use `DEBUG` for development and troubleshooting
- Use `INFO` for production monitoring
- Use `WARNING` for recoverable issues
- Use `ERROR` for failures that need attention

### 2. Log Rotation
- Enable log rotation for production environments
- Set appropriate retention periods
- Monitor disk space usage

### 3. Sensitive Data
- Be careful with debug logging in production
- Consider masking sensitive information in logs
- Use appropriate log levels to control data exposure

### 4. Performance
- Logging has minimal performance impact
- Debug logging can be verbose - use sparingly in production
- Consider async logging for high-throughput applications

### 5. Monitoring
- Set up alerts for ERROR level logs
- Monitor log file sizes and rotation
- Track usage patterns and costs through logs

## Troubleshooting

### Common Issues

1. **Logs not appearing**
   - Check log level configuration
   - Verify file permissions
   - Ensure log directory exists

2. **Too much debug output**
   - Increase log level to INFO or WARNING
   - Disable debug logging in production

3. **Log files too large**
   - Enable log rotation
   - Increase log level to reduce verbosity
   - Clean up old log files

4. **Missing structured logs**
   - Ensure structured logging is enabled
   - Check JSON formatting in log files

### Getting Help

If you encounter issues with logging:

1. Check the log files for error messages
2. Verify your logging configuration
3. Test with a simple example first
4. Check the examples in `examples/logging_example.py`

## API Reference

### Core Logging Functions

- `setup_logging(level, log_file, log_format)` - Basic logging setup
- `get_logger(name)` - Get a logger instance
- `LLMLoggingConfig` - Advanced logging configuration class

### Logging Functions

- `setup_logging()` - Basic logging configuration
- `get_logger()` - Get logger instances

### Logger Names

- `llm_utils` - Main application logger
- `llm_utils.tracker` - Usage tracking logger
- `llm_utils.client` - API client logger
- `llm_utils.errors` - Error-only logger
- `llm_utils.debug` - Debug-only logger
