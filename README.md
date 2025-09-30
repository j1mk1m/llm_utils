# LLM Utils

A comprehensive Python library for making LLM calls across different backends with built-in usage tracking, cost monitoring, and extensive logging capabilities.

## Features

- **Multi-Backend Support**: Works with OpenAI, Anthropic, AWS Bedrock, vLLM, and other providers via LiteLLM
- **Usage Tracking**: Automatic tracking of tokens, costs, and response times
- **Comprehensive Logging**: Built-in logging with support for prompts, responses, and error tracking
- **Data Persistence**: Save and load usage data in JSON or CSV formats
- **Cost Analytics**: Detailed cost breakdown by model and provider
- **Thread-Safe**: Safe for concurrent usage
- **Retry Logic**: Configurable retry with exponential backoff
- **Default Parameters**: Set default models and parameters for convenience

## Installation

```bash
pip install llm-utils
```

## Quick Start

### Basic Usage

```python
from llm_utils.llm_util import setup_logging, create_llm_client

# Set up logging
setup_logging(level="INFO", log_file="app.log")

# Create a client
client = create_llm_client(default_model="gpt-3.5-turbo")

# Make API calls
response = client.chat_completion([
    {"role": "user", "content": "Hello, world!"}
])

print(response['choices'][0]['message']['content'])
```

### With Usage Tracking

```python
from llm_utils.llm_util import create_llm_client, get_usage_summary

# Create client
client = create_llm_client(default_model="gpt-3.5-turbo")

# Make several API calls
for i in range(3):
    response = client.chat_completion([
        {"role": "user", "content": f"Tell me about topic {i+1}"}
    ])

# Get usage summary
stats = get_usage_summary(client)
print(f"Total requests: {stats['total_requests']}")
print(f"Total cost: {stats['total_cost']}")
print(f"Success rate: {stats['success_rate']}")
```

## Logging

The library includes comprehensive logging capabilities. See [LOGGING.md](LOGGING.md) for more details.

## API Reference

### Core Classes

#### `LLMClient`

Main client for making LLM requests.

```python
client = LLMClient(
    default_model="gpt-3.5-turbo",
    default_temperature=0.7,
    default_max_tokens=1000,
    default_retry_attempts=3,
    default_retry_delay=1.0
)
```

**Methods:**
- `chat_completion(messages, **kwargs)` - Chat completion requests
- `text_completion(prompt, **kwargs)` - Text completion requests
- `get_usage_stats(**kwargs)` - Get usage statistics
- `save_usage_data(file_path)` - Save usage data to file (includes checkpoints, usage aggregates)
- `load_usage_data(file_path)` - Load usage data from file (supports checkpoints, usage aggregates)
- `export_usage_data(file_path, format, checkpoint_name)` - Export usage data (only raw usage data)
- `start_usage_checkpoint(name)` - Start a usage checkpoint
- `end_usage_checkpoint(name)` - End a usage checkpoint
- `get_checkpoint_usage(name)` - Get checkpoint usage statistics

#### `LLMUsageTracker`

Tracks usage statistics and costs.

```python
tracker = LLMUsageTracker(data_file="usage.json")
```

**Methods:**
- `track_usage(...)` - Track a single usage event
- `get_aggregated_usage(...)` - Get aggregated statistics
- `save_usage_data(file_path)` - Save data to file (includes checkpoints)
- `load_usage_data(file_path)` - Load data from file (supports checkpoints)
- `export_usage_data(file_path, format, checkpoint_name)` - Export usage data
- `start_usage_checkpoint(name)` - Start a usage checkpoint
- `end_usage_checkpoint(name)` - End a usage checkpoint
- `get_checkpoint_usage(name)` - Get checkpoint usage statistics

### Convenience Functions

#### `create_llm_client()`

Create a new LLM client with usage tracking.

```python
client = create_llm_client(
    data_file="usage.json",
    default_model="gpt-3.5-turbo",
    default_temperature=0.7
)
```

#### `setup_logging()`

Set up logging configuration.

```python
logger = setup_logging(
    level="INFO",
    log_file="app.log",
    log_format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
```

#### `get_usage_summary()`

Get a summary of usage statistics.

```python
summary = get_usage_summary(client)
# Returns: {
#     "total_requests": 10,
#     "total_tokens": 1500,
#     "total_cost": "$0.0300",
#     "average_response_time": "1.2s",
#     "success_rate": "100.0%",
#     "top_models": [...],
#     "top_providers": [...]
# }
```

## Usage Tracking with checkpoints
```python
from llm_utils.llm_util import create_llm_client, get_usage_summary

# Create client
client = create_llm_client(default_model="gpt-3.5-turbo")

# First part 
client.start_usage_checkpoint("first")
for i in range(3):
    response = client.chat_completion([
        {"role": "user", "content": f"Tell me about topic {i+1}"}
    ])
client.end_usage_checkpoint("first")

# Second part (with nested checkpointing)
client.start_usage_checkpoint("second")
for i in range(3):
    client.start_usage_checkpoint(f"second_loop_iteration_{i+1}")
    response = client.chat_completion([
        {"role": "user", "content": f"Tell me again about topic {i+1}"}
    ])
    client.end_usage_checkpoint(f"second_loop_iteration_{i+1}")
client.end_usage_checkpoint("second")

# Get usage summary
stats = client.get_checkpoint_usage("first")
print(f"Total requests for first part: {stats['total_requests']}")
stats = client.get_checkpoint_usage("second")
print(f"Total requests for second part: {stats['total_requests']}")
stats = client.get_checkpoint_usage("second_loop_iteration_1")
print(f"Total requests for second part iteration 1: {stats['total_requests']}")
```

### Checkpoint Data Management

The library provides comprehensive data management for checkpoints:

#### Saving Checkpoint Data

```python
# Save all usage data including checkpoint information
client.save_usage_data("usage_with_checkpoints.json")

# The saved file includes:
# - usage_data: All usage records
# - checkpoint_ranges: Checkpoint index ranges
# - checkpoint_stacks: Active checkpoint stacks
# - metadata: Save timestamp and statistics
```

#### Exporting Checkpoint Data

```python
# Export all usage data
client.export_usage_data("all_usage.json", format="json")

# Export data for a specific checkpoint
client.export_usage_data("first_checkpoint.json", format="json", checkpoint_name="first")
client.export_usage_data("second_checkpoint.csv", format="csv", checkpoint_name="second")

# Export nested checkpoint data
client.export_usage_data("iteration_data.json", format="json", checkpoint_name="second_loop_iteration_1")
```

#### Loading Checkpoint Data

```python
# Load data with checkpoint information (new format)
client.load_usage_data("usage_with_checkpoints.json")

# Load legacy format data (backward compatible)
client.load_usage_data("legacy_usage.json")

# After loading, checkpoints are restored and available
stats = client.get_checkpoint_usage("first")  # Works after loading
```

#### Checkpoint Data Structure

The saved data structure includes:

```json
{
  "usage_aggregate": {

  },
  "usage_data": [
    {
      "timestamp": "2024-01-01T12:00:00Z",
      "model": "gpt-3.5-turbo",
      "provider": "openai",
      "prompt_tokens": 10,
      "completion_tokens": 5,
      "total_tokens": 15,
      "cost": 0.001,
      "response_time": 1.0,
      "success": true
    }
  ],
  "checkpoint_ranges": {
    "first": [[1, 3]],
    "second": [[3, 6]]
  },
  "checkpoint_stacks": {},
  "checkpoint_aggregates": {

  },
  "metadata": {
    "saved_at": "2024-01-01T12:30:00Z",
    "total_usage_records": 5,
    "total_checkpoints": 2
  }
}
```

## Usage Examples
See [examples/basic_usage.py](examples/basic_usage.py) for examples of basic usage.


## Configuration

### Environment Variables

Set your API keys as environment variables:

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export AWS_ACCESS_KEY_ID="your-aws-key"
export AWS_SECRET_ACCESS_KEY="your-aws-secret"
```

### Default Parameters

You can set default parameters when creating a client:

```python
client = create_llm_client(
    default_model="gpt-3.5-turbo",
    default_temperature=0.7,
    default_max_tokens=1000,
    default_api_base="https://api.openai.com/v1",
    default_retry_attempts=2,
    default_retry_delay=1.0
)
```

### Retry Configuration

Configure retry behavior:

```python
client = LLMClient(
    default_retry_attempts=5,  # Number of retry attempts
    default_retry_delay=2.0    # Base delay between retries (exponential backoff)
)
```

## Supported Providers

The library supports all providers supported by [LiteLLM](https://docs.litellm.ai/).

## Testing

Run the test suite:

```bash
python -m unittest tests.test_llm_utils
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Changelog

### v1.0.0
- Initial release
- Multi-backend LLM support via LiteLLM
- Usage tracking and cost monitoring
- Comprehensive logging system
- Data persistence (JSON/CSV)
- Thread-safe operations
- Configurable retry logic
- Usage checkpoints for tracking specific code sections
- Checkpoint-aware data export and import
