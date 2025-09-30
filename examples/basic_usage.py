#!/usr/bin/env python3
"""
Comprehensive basic usage examples for llm_utils

This script demonstrates end-to-end usage of the llm_utils package, including:
- Logging setup
- Creating an LLM client
- Chat and text completions
- Overriding default parameters
- Usage summaries and analytics
- Checkpoints (start/end, nested), and retrieving checkpoint usage
- Data export (JSON/CSV) and load
- Basic error handling patterns

Note: Running API calls requires appropriate provider API keys set as env vars
(e.g., OPENAI_API_KEY). See README for details.
"""

import os
from datetime import datetime, timezone

from llm_utils import (
    setup_logging,
    create_llm_client,
    get_usage_summary,
)


def example_setup_and_client():
    print("=== Setup and Client Creation ===")
    setup_logging(level="INFO", log_file="basic_usage.log")

    client = create_llm_client(
        default_model="gpt-3.5-turbo",
        default_temperature=0.7,
        default_max_tokens=512,
        default_retry_attempts=3,
        default_retry_delay=1.0,
    )

    print("Client created with defaults. Logs written to basic_usage.log\n")
    return client


def example_chat_completion(client):
    print("=== Chat Completion ===")
    response = client.chat_completion(
        [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "Explain RAG in one paragraph."},
        ]
    )
    print(response["choices"][0]["message"]["content"].strip(), "\n")


def example_text_completion(client):
    print("=== Text Completion ===")
    response = client.text_completion(
        prompt="Write a short haiku about refactoring code.",
        temperature=0.8,
        max_tokens=120,
    )
    print(response["choices"][0]["text"].strip(), "\n")


def example_parameter_overrides(client):
    print("=== Parameter Overrides ===")
    response = client.chat_completion(
        messages=[{"role": "user", "content": "List 3 Python testing frameworks."}],
        model="gpt-4",
        temperature=0.3,
        max_tokens=150,
    )
    print(response["choices"][0]["message"]["content"].strip(), "\n")


def example_usage_summary(client):
    print("=== Usage Summary ===")
    summary = get_usage_summary(client)
    print(f"Total requests: {summary['total_requests']}")
    print(f"Total tokens: {summary.get('total_tokens')}")
    print(f"Total cost: {summary['total_cost']}")
    print(f"Avg response time: {summary.get('average_response_time')}")
    print(f"Success rate: {summary['success_rate']}\n")


def example_usage_analytics(client):
    print("=== Usage Analytics (Date/Model Filters) ===")
    start_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end_date = datetime(2026, 1, 1, tzinfo=timezone.utc)
    stats = client.get_usage_stats(start_date=start_date, end_date=end_date)
    print(f"Requests in range: {stats.total_requests}")
    gpt4_stats = client.get_usage_stats(model_filter="gpt-4")
    print(f"Requests for gpt-4: {gpt4_stats.total_requests}\n")


def example_checkpoints(client):
    print("=== Usage Checkpoints (including nested) ===")

    client.start_usage_checkpoint("phase_one")
    _ = client.chat_completion([{"role": "user", "content": "One fun fact about turtles."}])
    _ = client.chat_completion([{"role": "user", "content": "One fun fact about owls."}])
    client.end_usage_checkpoint("phase_one")

    client.start_usage_checkpoint("phase_two")
    for i in range(2):
        name = f"phase_two_iter_{i+1}"
        client.start_usage_checkpoint(name)
        _ = client.chat_completion([
            {"role": "user", "content": f"Give me a coding tip #{i+1}."}
        ])
        client.end_usage_checkpoint(name)
    client.end_usage_checkpoint("phase_two")

    phase_one_stats = client.get_checkpoint_usage("phase_one")
    phase_two_stats = client.get_checkpoint_usage("phase_two")
    iter1_stats = client.get_checkpoint_usage("phase_two_iter_1")

    print(f"phase_one total requests: {phase_one_stats['total_requests']}")
    print(f"phase_two total requests: {phase_two_stats['total_requests']}")
    print(f"phase_two_iter_1 total requests: {iter1_stats['total_requests']}\n")


def example_export_and_load(client):
    print("=== Export and Load Usage Data ===")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = f"usage_{timestamp}.json"
    csv_path = f"usage_{timestamp}.csv"

    client.save_usage_data(json_path)
    client.export_usage_data(csv_path, format="csv")

    # Demonstrate loading back
    client.load_usage_data(json_path)
    restored = client.get_usage_stats()
    print("Usage data saved and reloaded.")
    print(f"Restored total requests: {restored.total_requests}")
    print(f"Exported files: {json_path}, {csv_path}\n")


def example_error_handling():
    print("=== Error Handling Pattern ===")
    client = create_llm_client(default_model="gpt-3.5-turbo", default_retry_attempts=2)
    try:
        _ = client.chat_completion([{"role": "user", "content": "Hello"}])
        print("Request succeeded.\n")
    except Exception as exc:
        # In real apps, prefer logging and structured error handling
        print(f"API call failed: {exc}\n")


def ensure_env_keys_hint():
    # Lightweight hint for users who run this without API keys configured
    missing = []
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        if not os.environ.get(key):
            missing.append(key)
    if missing:
        print("Note: Missing potential provider credentials:")
        for k in missing:
            print(f"- {k} not set")
        print("Some requests may fail without valid credentials.\n")


def main():
    ensure_env_keys_hint()

    client = example_setup_and_client()

    # Core flows
    example_chat_completion(client)
    example_text_completion(client)
    example_parameter_overrides(client)

    # Usage views
    example_usage_summary(client)
    example_usage_analytics(client)

    # Checkpoints and data management
    example_checkpoints(client)
    example_export_and_load(client)

    # Error handling pattern
    example_error_handling()

    print("All basic usage examples executed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Unexpected error in basic usage examples: {e}")
        import traceback
        traceback.print_exc()


