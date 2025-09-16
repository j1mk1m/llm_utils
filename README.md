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

The library includes comprehensive logging capabilities:

### Basic Logging Setup

```python
from llm_utils.llm_util import setup_logging

# Console logging only
setup_logging(level="INFO")

# File logging
setup_logging(level="INFO", log_file="app.log")

# Debug logging (includes prompts and responses)
setup_logging(level="DEBUG", log_file="debug.log")
```

### Log Levels

- **INFO**: API calls, usage tracking, successful operations
- **DEBUG**: Full prompts, responses, detailed parameters
- **WARNING**: Retry attempts, recoverable errors
- **ERROR**: Failed API calls, unexpected errors

### Example Log Output

```
2024-01-01 12:00:00 - llm_utils.client - INFO - Making LLM request to gpt-3.5-turbo (openai)
2024-01-01 12:00:00 - llm_utils.client - DEBUG - Request messages: [{'role': 'user', 'content': 'Hello'}]
2024-01-01 12:00:00 - llm_utils.tracker - INFO - Tracked successful usage: gpt-3.5-turbo (openai) - tokens: 15, cost: $0.0010, response_time: 0.50s
2024-01-01 12:00:00 - llm_utils.client - INFO - LLM request successful: gpt-3.5-turbo - tokens: 15, cost: $0.0010, response_time: 0.50s
```

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
- `save_usage_data(file_path)` - Save usage data to file
- `load_usage_data(file_path)` - Load usage data from file

#### `LLMUsageTracker`

Tracks usage statistics and costs.

```python
tracker = LLMUsageTracker(data_file="usage.json")
```

**Methods:**
- `track_usage(...)` - Track a single usage event
- `get_aggregated_usage(...)` - Get aggregated statistics
- `save_usage_data(file_path)` - Save data to file
- `load_usage_data(file_path)` - Load data from file

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

## Usage Examples

### Chat Completion

```python
from llm_utils.llm_util import setup_logging, create_llm_client

# Set up logging
setup_logging(level="INFO", log_file="chat.log")

# Create client
client = create_llm_client(default_model="gpt-4")

# Make chat completion request
response = client.chat_completion([
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain quantum computing in simple terms."}
])

print(response['choices'][0]['message']['content'])
```

### Text Completion

```python
# Text completion request
response = client.text_completion(
    prompt="Write a short poem about coding",
    temperature=0.9,
    max_tokens=200
)

print(response['choices'][0]['message']['content'])
```

### Custom Parameters

```python
# Override default parameters
response = client.chat_completion(
    messages=[{"role": "user", "content": "Hello"}],
    model="gpt-4",  # Override default model
    temperature=0.5,  # Override default temperature
    max_tokens=500   # Override default max_tokens
)
```

### Usage Analytics

```python
# Get detailed usage statistics
stats = client.get_usage_stats()

print(f"Total requests: {stats.total_requests}")
print(f"Total tokens: {stats.total_tokens}")
print(f"Total cost: ${stats.total_cost:.4f}")
print(f"Average response time: {stats.average_response_time:.2f}s")
print(f"Success rate: {stats.success_rate:.1%}")

# Filter by date range
from datetime import datetime, timezone
start_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
end_date = datetime(2024, 1, 31, tzinfo=timezone.utc)

monthly_stats = client.get_usage_stats(
    start_date=start_date,
    end_date=end_date
)

# Filter by model
gpt4_stats = client.get_usage_stats(model_filter="gpt-4")
```

### Data Export

```python
# Save usage data
client.save_usage_data("usage_backup.json")

# Export to CSV
client.export_usage_data("usage_report.csv", format="csv")

# Load historical data
client.load_usage_data("historical_usage.json")
```

### Error Handling

```python
from llm_utils.llm_util import setup_logging, create_llm_client

# Set up logging to see error details
setup_logging(level="INFO", log_file="error.log")

client = create_llm_client(
    default_model="gpt-3.5-turbo",
    default_retry_attempts=3,
    default_retry_delay=1.0
)

try:
    response = client.chat_completion([
        {"role": "user", "content": "Hello"}
    ])
except Exception as e:
    print(f"API call failed: {e}")
    # Check error.log for detailed error information
```

### Debug Mode

```python
# Enable debug logging to see prompts and responses
setup_logging(level="DEBUG", log_file="debug.log")

client = create_llm_client(default_model="gpt-3.5-turbo")

# This will log the full prompt and response
response = client.chat_completion([
    {"role": "user", "content": "Tell me a joke"}
])
```

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
    default_api_base="https://api.openai.com/v1"
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

The library supports all providers supported by LiteLLM:

- **OpenAI**: GPT-3.5, GPT-4, etc.
- **Anthropic**: Claude-3, Claude-2, etc.
- **AWS Bedrock**: Various models
- **vLLM**: Local and remote vLLM servers
- **Google**: PaLM, Gemini models
- **Cohere**: Command models
- **Hugging Face**: Various models
- And many more...

## Logging Documentation

For detailed logging configuration and examples, see [LOGGING.md](LOGGING.md).

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
