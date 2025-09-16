"""
LiteLLM Utility Module for Multi-Backend LLM Calls with Usage Tracking

This module provides a comprehensive interface for making LLM calls across different
backends (AWS Bedrock, vLLM, OpenAI, Anthropic, etc.) using LiteLLM, with built-in
usage tracking and data persistence capabilities.

Features:
- Multi-backend support (OpenAI, Anthropic, AWS Bedrock, vLLM, etc.)
- Automatic usage tracking and aggregation
- Data persistence (save/load usage data)
- Cost calculation and analytics
- Thread-safe operations
- Configurable retry logic
"""

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Literal
from dataclasses import dataclass, asdict
from collections import defaultdict

from litellm import completion, get_llm_provider, token_counter, cost_per_token
from litellm import completion_cost as litellm_get_total_cost
from litellm.exceptions import RateLimitError, APIConnectionError, APIError


# Configure logging
def setup_logging(level: str = "INFO", 
                 log_file: Optional[str] = None,
                 log_format: Optional[str] = None) -> logging.Logger:
    """
    Set up logging configuration for the LLM utilities.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to write logs to
        log_format: Optional custom log format string
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("llm_utils")
    
    # Clear any existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Set logging level
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)
    
    # Default log format
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    formatter = logging.Formatter(log_format)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def get_logger(name: str = "llm_utils") -> logging.Logger:
    """
    Get a logger instance for the specified name.
    
    Args:
        name: Logger name (defaults to "llm_utils")
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# Initialize default logger
logger = get_logger()


@dataclass
class UsageData:
    """Data class for tracking LLM usage information."""
    timestamp: str
    model: str
    provider: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float
    request_id: Optional[str] = None
    response_time: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    prompt_cost: Optional[float] = None
    completion_cost: Optional[float] = None


@dataclass
class UsageAggregate:
    """Data class for aggregated usage statistics."""
    total_requests: int
    total_tokens: int
    total_cost: float
    total_prompt_tokens: int
    total_completion_tokens: int
    average_response_time: float
    success_rate: float
    model_breakdown: Dict[str, Dict[str, Any]]
    provider_breakdown: Dict[str, Dict[str, Any]]
    time_range: Dict[str, str]


class LLMUsageTracker:
    """
    A comprehensive LLM usage tracker that integrates with LiteLLM.
    
    This class provides functionality to:
    - Track usage across multiple LLM backends
    - Aggregate usage statistics
    - Persist usage data to files
    - Load historical usage data
    - Calculate costs and analytics
    """
    
    def __init__(self, data_file: Optional[str] = None, logger: Optional[logging.Logger] = None):
        """
        Initialize the usage tracker.
        
        Args:
            data_file: Path to the JSON file for persisting usage data.
                      If None, defaults to 'llm_usage_data.json' in current directory.
            logger: Optional logger instance. If None, uses default logger.
        """
        self.data_file = data_file or "llm_usage_data.json"
        self.usage_data: List[UsageData] = []
        self.lock = threading.Lock()
        self.logger = logger or get_logger("llm_utils.tracker")
        # Checkpoint tracking structures
        self._checkpoint_stacks: Dict[str, List[int]] = defaultdict(list)
        self._checkpoint_ranges: Dict[str, List[tuple[int, int]]] = defaultdict(list)
        
        # Load existing data if file exists
        self.load_usage_data()
    
    def calculate_token_costs(self, model: str, prompt_tokens: int, completion_tokens: int) -> tuple[float, float, float]:
        """
        Calculate detailed costs for prompt and completion tokens.
        
        Args:
            model: Model name
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            
        Returns:
            Tuple of (prompt_cost, completion_cost, total_cost)
        """
        try:
            prompt_cost, completion_cost = cost_per_token(
                model=model, 
                prompt_tokens=prompt_tokens, 
                completion_tokens=completion_tokens
            )
            total_cost = prompt_cost + completion_cost
            return prompt_cost, completion_cost, total_cost
        except Exception as e:
            self.logger.warning(f"Failed to calculate costs for {model}: {e}")
            return 0.0, 0.0, 0.0
    
    def track_usage(self, 
                   model: str,
                   provider: str,
                   prompt_tokens: int,
                   completion_tokens: int,
                   total_tokens: int,
                   cost: float,
                   request_id: Optional[str] = None,
                   response_time: Optional[float] = None,
                   success: bool = True,
                   error_message: Optional[str] = None,
                   prompt_cost: Optional[float] = None,
                   completion_cost: Optional[float] = None) -> None:
        """
        Track a single LLM usage event.
        
        Args:
            model: The model name used
            provider: The provider (e.g., 'openai', 'anthropic', 'aws_bedrock')
            prompt_tokens: Number of tokens in the prompt
            completion_tokens: Number of tokens in the completion
            total_tokens: Total tokens used
            cost: Cost of the request
            request_id: Optional request ID
            response_time: Response time in seconds
            success: Whether the request was successful
            error_message: Error message if request failed
            prompt_cost: Cost for prompt tokens only
            completion_cost: Cost for completion tokens only
        """
        usage = UsageData(
            timestamp=datetime.now(timezone.utc).isoformat(),
            model=model,
            provider=provider,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost=cost,
            request_id=request_id,
            response_time=response_time,
            success=success,
            error_message=error_message,
            prompt_cost=prompt_cost,
            completion_cost=completion_cost,
        )
        
        with self.lock:
            self.usage_data.append(usage)
        
        # Log usage tracking
        if success:
            response_time_str = f"{response_time:.2f}s" if response_time is not None else "N/A"
            self.logger.info(
                f"Tracked successful usage: {model} ({provider}) - "
                f"tokens: {total_tokens}, cost: ${cost:.4f}, "
                f"response_time: {response_time_str}" + 
                (f", request_id: {request_id}" if request_id else "")
            )
        else:
            response_time_str = f"{response_time:.2f}s" if response_time is not None else "N/A"
            self.logger.error(
                f"Tracked failed usage: {model} ({provider}) - "
                f"error: {error_message}, response_time: {response_time_str}" +
                (f", request_id: {request_id}" if request_id else "")
            )
    
    def get_aggregated_usage(self, 
                           start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None,
                           model_filter: Optional[str] = None,
                           provider_filter: Optional[str] = None) -> UsageAggregate:
        """
        Get aggregated usage statistics.
        
        Args:
            start_date: Filter data from this date onwards
            end_date: Filter data up to this date
            model_filter: Filter by specific model
            provider_filter: Filter by specific provider
            
        Returns:
            UsageAggregate object with statistics
        """
        with self.lock:
            filtered_data = self.usage_data.copy()
        
        # Apply filters
        if start_date:
            filtered_data = [d for d in filtered_data if datetime.fromisoformat(d.timestamp.replace('Z', '+00:00')) >= start_date]
        if end_date:
            filtered_data = [d for d in filtered_data if datetime.fromisoformat(d.timestamp.replace('Z', '+00:00')) <= end_date]
        if model_filter:
            filtered_data = [d for d in filtered_data if d.model == model_filter]
        if provider_filter:
            filtered_data = [d for d in filtered_data if d.provider == provider_filter]
        
        if not filtered_data:
            return UsageAggregate(
                total_requests=0,
                total_tokens=0,
                total_cost=0.0,
                total_prompt_tokens=0,
                total_completion_tokens=0,
                average_response_time=0.0,
                success_rate=0.0,
                model_breakdown={},
                provider_breakdown={},
                time_range={"start": "", "end": ""}
            )
        
        # Calculate basic statistics
        total_requests = len(filtered_data)
        total_tokens = sum(d.total_tokens for d in filtered_data)
        total_cost = sum(d.cost for d in filtered_data)
        total_prompt_tokens = sum(d.prompt_tokens for d in filtered_data)
        total_completion_tokens = sum(d.completion_tokens for d in filtered_data)
        
        # Calculate response time statistics
        response_times = [d.response_time for d in filtered_data if d.response_time is not None]
        average_response_time = sum(response_times) / len(response_times) if response_times else 0.0
        
        # Calculate success rate
        successful_requests = sum(1 for d in filtered_data if d.success)
        success_rate = successful_requests / total_requests if total_requests > 0 else 0.0
        
        # Model breakdown
        model_breakdown = defaultdict(lambda: {
            'requests': 0, 'tokens': 0, 'cost': 0.0, 'avg_response_time': 0.0
        })
        for data in filtered_data:
            model_breakdown[data.model]['requests'] += 1
            model_breakdown[data.model]['tokens'] += data.total_tokens
            model_breakdown[data.model]['cost'] += data.cost
            if data.response_time is not None:
                model_breakdown[data.model]['avg_response_time'] += data.response_time
        
        # Calculate averages for model breakdown
        for model_data in model_breakdown.values():
            if model_data['requests'] > 0:
                model_data['avg_response_time'] /= model_data['requests']
        
        # Provider breakdown
        provider_breakdown = defaultdict(lambda: {
            'requests': 0, 'tokens': 0, 'cost': 0.0, 'avg_response_time': 0.0
        })
        for data in filtered_data:
            provider_breakdown[data.provider]['requests'] += 1
            provider_breakdown[data.provider]['tokens'] += data.total_tokens
            provider_breakdown[data.provider]['cost'] += data.cost
            if data.response_time is not None:
                provider_breakdown[data.provider]['avg_response_time'] += data.response_time
        
        # Calculate averages for provider breakdown
        for provider_data in provider_breakdown.values():
            if provider_data['requests'] > 0:
                provider_data['avg_response_time'] /= provider_data['requests']
        
        # Time range
        timestamps = [datetime.fromisoformat(d.timestamp.replace('Z', '+00:00')) for d in filtered_data]
        time_range = {
            "start": min(timestamps).isoformat() if timestamps else "",
            "end": max(timestamps).isoformat() if timestamps else ""
        }
        
        return UsageAggregate(
            total_requests=total_requests,
            total_tokens=total_tokens,
            total_cost=total_cost,
            total_prompt_tokens=total_prompt_tokens,
            total_completion_tokens=total_completion_tokens,
            average_response_time=average_response_time,
            success_rate=success_rate,
            model_breakdown=dict(model_breakdown),
            provider_breakdown=dict(provider_breakdown),
            time_range=time_range
        )

    def start_usage_checkpoint(self, name: str) -> None:
        """
        Start a usage checkpoint identified by name.

        Supports nested checkpoints by maintaining a stack per name.
        """
        with self.lock:
            self._checkpoint_stacks[name].append(len(self.usage_data))
            self.logger.debug(f"Started usage checkpoint '{name}' at index {self._checkpoint_stacks[name][-1]}")

    def end_usage_checkpoint(self, name: str) -> None:
        """
        End a usage checkpoint identified by name.

        Records the range [start_index, end_index) of usage events for this checkpoint.
        """
        with self.lock:
            if name not in self._checkpoint_stacks or not self._checkpoint_stacks[name]:
                raise ValueError(f"No active checkpoint to end for name '{name}'")
            start_index = self._checkpoint_stacks[name].pop()
            end_index = len(self.usage_data)
            self._checkpoint_ranges[name].append((start_index, end_index))
            self.logger.debug(f"Ended usage checkpoint '{name}' from {start_index} to {end_index}")

    def get_checkpoint_usage(self, name: str) -> UsageAggregate:
        """
        Get aggregated usage statistics for a named checkpoint.

        If multiple intervals were recorded for the checkpoint name, aggregates
        across the union of those intervals (deduplicated indices).
        """
        with self.lock:
            ranges = list(self._checkpoint_ranges.get(name, []))
            # If there is an open (not yet ended) checkpoint for this name,
            # include it up to the current end of usage_data.
            if name in self._checkpoint_stacks and self._checkpoint_stacks[name]:
                for start_index in self._checkpoint_stacks[name]:
                    ranges.append((start_index, len(self.usage_data)))

            if not ranges:
                return UsageAggregate(
                    total_requests=0,
                    total_tokens=0,
                    total_cost=0.0,
                    total_prompt_tokens=0,
                    total_completion_tokens=0,
                    average_response_time=0.0,
                    success_rate=0.0,
                    model_breakdown={},
                    provider_breakdown={},
                    time_range={"start": "", "end": ""}
                )

            # Build a deduplicated set of indices covered by any interval
            covered_indices = set()
            for start_idx, end_idx in ranges:
                if start_idx < 0:
                    start_idx = 0
                if end_idx < 0:
                    end_idx = 0
                if start_idx > end_idx:
                    start_idx, end_idx = end_idx, start_idx
                covered_indices.update(range(start_idx, min(end_idx, len(self.usage_data))))

            filtered_data = [self.usage_data[i] for i in sorted(covered_indices)]

        # Reuse aggregation logic on the filtered_data (outside lock)
        if not filtered_data:
            return UsageAggregate(
                total_requests=0,
                total_tokens=0,
                total_cost=0.0,
                total_prompt_tokens=0,
                total_completion_tokens=0,
                average_response_time=0.0,
                success_rate=0.0,
                model_breakdown={},
                provider_breakdown={},
                time_range={"start": "", "end": ""}
            )

        total_requests = len(filtered_data)
        total_tokens = sum(d.total_tokens for d in filtered_data)
        total_cost = sum(d.cost for d in filtered_data)
        total_prompt_tokens = sum(d.prompt_tokens for d in filtered_data)
        total_completion_tokens = sum(d.completion_tokens for d in filtered_data)

        response_times = [d.response_time for d in filtered_data if d.response_time is not None]
        average_response_time = sum(response_times) / len(response_times) if response_times else 0.0

        successful_requests = sum(1 for d in filtered_data if d.success)
        success_rate = successful_requests / total_requests if total_requests > 0 else 0.0

        model_breakdown = defaultdict(lambda: {
            'requests': 0, 'tokens': 0, 'cost': 0.0, 'avg_response_time': 0.0
        })
        for data in filtered_data:
            model_breakdown[data.model]['requests'] += 1
            model_breakdown[data.model]['tokens'] += data.total_tokens
            model_breakdown[data.model]['cost'] += data.cost
            if data.response_time is not None:
                model_breakdown[data.model]['avg_response_time'] += data.response_time
        for model_data in model_breakdown.values():
            if model_data['requests'] > 0:
                model_data['avg_response_time'] /= model_data['requests']

        provider_breakdown = defaultdict(lambda: {
            'requests': 0, 'tokens': 0, 'cost': 0.0, 'avg_response_time': 0.0
        })
        for data in filtered_data:
            provider_breakdown[data.provider]['requests'] += 1
            provider_breakdown[data.provider]['tokens'] += data.total_tokens
            provider_breakdown[data.provider]['cost'] += data.cost
            if data.response_time is not None:
                provider_breakdown[data.provider]['avg_response_time'] += data.response_time
        for provider_data in provider_breakdown.values():
            if provider_data['requests'] > 0:
                provider_data['avg_response_time'] /= provider_data['requests']

        timestamps = [datetime.fromisoformat(d.timestamp.replace('Z', '+00:00')) for d in filtered_data]
        time_range = {
            "start": min(timestamps).isoformat() if timestamps else "",
            "end": max(timestamps).isoformat() if timestamps else ""
        }

        return UsageAggregate(
            total_requests=total_requests,
            total_tokens=total_tokens,
            total_cost=total_cost,
            total_prompt_tokens=total_prompt_tokens,
            total_completion_tokens=total_completion_tokens,
            average_response_time=average_response_time,
            success_rate=success_rate,
            model_breakdown=dict(model_breakdown),
            provider_breakdown=dict(provider_breakdown),
            time_range=time_range
        )
    
    def save_usage_data(self, file_path: Optional[str] = None) -> None:
        """
        Save usage data to a JSON file including checkpoint information.
        
        Args:
            file_path: Optional custom file path. If None, uses the default data_file.
        """
        save_path = file_path or self.data_file
        
        with self.lock:
            data_to_save = {
                "usage_data": [asdict(usage) for usage in self.usage_data],
                "checkpoint_ranges": dict(self._checkpoint_ranges),
                "checkpoint_stacks": dict(self._checkpoint_stacks),
                "metadata": {
                    "saved_at": datetime.now(timezone.utc).isoformat(),
                    "total_usage_records": len(self.usage_data),
                    "total_checkpoints": len(self._checkpoint_ranges)
                }
            }
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        try:
            with open(save_path, 'w') as f:
                json.dump(data_to_save, f, indent=2)
            self.logger.info(f"Saved {len(self.usage_data)} usage records and {len(self._checkpoint_ranges)} checkpoints to {save_path}")
        except Exception as e:
            self.logger.error(f"Failed to save usage data to {save_path}: {e}")
            raise
    
    def load_usage_data(self, file_path: Optional[str] = None) -> None:
        """
        Load usage data from a JSON file including checkpoint information.
        
        Args:
            file_path: Optional custom file path. If None, uses the default data_file.
        """
        load_path = file_path or self.data_file
        
        if not os.path.exists(load_path):
            self.logger.debug(f"Usage data file {load_path} does not exist, starting with empty data")
            return
        
        try:
            with open(load_path, 'r') as f:
                data = json.load(f)
            
            with self.lock:
                # Handle both old format (list) and new format (dict with checkpoint info)
                if isinstance(data, list):
                    # Old format - just usage data
                    self.usage_data = [UsageData(**item) for item in data]
                    self.logger.info(f"Loaded {len(self.usage_data)} usage records from {load_path} (legacy format)")
                elif isinstance(data, dict) and "usage_data" in data:
                    # New format - includes checkpoint information
                    self.usage_data = [UsageData(**item) for item in data["usage_data"]]
                    
                    # Restore checkpoint information
                    self._checkpoint_ranges = defaultdict(list, data.get("checkpoint_ranges", {}))
                    self._checkpoint_stacks = defaultdict(list, data.get("checkpoint_stacks", {}))
                    
                    # Convert checkpoint_ranges keys back to tuples
                    for name, ranges in self._checkpoint_ranges.items():
                        self._checkpoint_ranges[name] = [tuple(r) for r in ranges]
                    
                    metadata = data.get("metadata", {})
                    self.logger.info(
                        f"Loaded {len(self.usage_data)} usage records and {len(self._checkpoint_ranges)} checkpoints from {load_path}"
                    )
                    if "saved_at" in metadata:
                        self.logger.debug(f"Data was saved at: {metadata['saved_at']}")
                else:
                    raise ValueError("Invalid data format in file")
                    
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            self.logger.warning(f"Could not load usage data from {load_path}: {e}")
            # Keep existing data if loading fails
    
    def clear_usage_data(self) -> None:
        """Clear all usage data from memory."""
        with self.lock:
            self.usage_data.clear()
    
    def export_usage_data(self, file_path: str, format: Literal['json', 'csv'] = 'json', checkpoint_name: Optional[str] = None) -> None:
        """
        Export usage data in different formats.
        
        Args:
            file_path: Path to save the exported data
            format: Export format ('json' or 'csv')
            checkpoint_name: Optional checkpoint name to export only data from that checkpoint
        """
        with self.lock:
            if checkpoint_name:
                # Get data for specific checkpoint
                checkpoint_data = self._get_checkpoint_data(checkpoint_name)
                data_to_export = [asdict(usage) for usage in checkpoint_data]
                self.logger.info(f"Exporting {len(data_to_export)} usage records from checkpoint '{checkpoint_name}'")
            else:
                # Export all data
                data_to_export = [asdict(usage) for usage in self.usage_data]
                self.logger.info(f"Exporting {len(data_to_export)} usage records")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        if format == 'json':
            with open(file_path, 'w') as f:
                json.dump(data_to_export, f, indent=2)
        elif format == 'csv':
            import csv
            if data_to_export:
                with open(file_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=data_to_export[0].keys())
                    writer.writeheader()
                    writer.writerows(data_to_export)
    
    def _get_checkpoint_data(self, checkpoint_name: str) -> List[UsageData]:
        """
        Get usage data for a specific checkpoint.
        
        Args:
            checkpoint_name: Name of the checkpoint
            
        Returns:
            List of UsageData objects for the checkpoint
        """
        ranges = list(self._checkpoint_ranges.get(checkpoint_name, []))
        # If there is an open (not yet ended) checkpoint for this name,
        # include it up to the current end of usage_data.
        if checkpoint_name in self._checkpoint_stacks and self._checkpoint_stacks[checkpoint_name]:
            for start_index in self._checkpoint_stacks[checkpoint_name]:
                ranges.append((start_index, len(self.usage_data)))
        
        if not ranges:
            return []
        
        # Collect all indices covered by the ranges
        indices = set()
        for start, end in ranges:
            indices.update(range(start, end))
        
        # Return usage data for those indices
        return [self.usage_data[i] for i in sorted(indices) if i < len(self.usage_data)]


class LLMClient:
    """
    A comprehensive LLM client that supports multiple backends with usage tracking.
    
    This class provides a unified interface for making LLM calls across different
    providers while automatically tracking usage and costs.
    """
    
    def __init__(self, 
                 usage_tracker: Optional[LLMUsageTracker] = None, 
                 default_retry_attempts: int = 3, 
                 default_retry_delay: float = 1.0,
                 default_model: Optional[str] = None,
                 default_temperature: Optional[float] = None,
                 default_max_tokens: Optional[int] = None,
                 default_api_base: Optional[str] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the LLM client.
        
        Args:
            usage_tracker: Optional usage tracker instance. If None, creates a new one.
            default_retry_attempts: Number of retry attempts for API errors
            default_retry_delay: Delay between retry attempts in seconds
            default_model: Default model to use for requests
            default_temperature: Default temperature for requests
            default_max_tokens: Default max_tokens for requests
            default_api_base: Default API base URL for requests
            logger: Optional logger instance. If None, uses default logger.
        """
        self.usage_tracker = usage_tracker or LLMUsageTracker()
        self.default_retry_attempts = default_retry_attempts
        self.default_retry_delay = default_retry_delay
        self.default_model = default_model
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self.default_api_base = default_api_base
        self.logger = logger or get_logger("llm_utils.client")
    
    def _make_request(self, 
                     messages: List[Dict[str, str]],
                     model: Optional[str] = None,
                     **kwargs) -> Dict[str, Any]:
        """
        Make an LLM request with retry logic and usage tracking.
        
        Args:
            messages: List of message dictionaries
            model: The model to use (e.g., 'gpt-4', 'claude-3-sonnet'). If None, uses default_model.
            **kwargs: Additional arguments for the completion call
            
        Returns:
            Response dictionary from LiteLLM
        """
        start_time = time.time()
        request_id = kwargs.get('request_id')
        
        # Use default model if not provided
        if model is None:
            if self.default_model is None:
                raise ValueError("No model provided and no default_model set")
            model = self.default_model
        
        # Merge default parameters with provided kwargs
        merged_kwargs = {}
        
        # Add default parameters if they exist and are not overridden
        if self.default_temperature is not None and 'temperature' not in kwargs:
            merged_kwargs['temperature'] = self.default_temperature
        if self.default_max_tokens is not None and 'max_tokens' not in kwargs:
            merged_kwargs['max_tokens'] = self.default_max_tokens
        if self.default_api_base is not None and 'api_base' not in kwargs:
            merged_kwargs['api_base'] = self.default_api_base
        
        # Add all provided kwargs (these will override defaults)
        merged_kwargs.update(kwargs)
        
        # Get provider information
        try:
            provider = get_llm_provider(model)
        except Exception:
            provider = "unknown"
        
        # Log request details
        self.logger.info(f"Making LLM request to {model} ({provider})" + 
                        (f" with request_id: {request_id}" if request_id else ""))
        
        # Log prompts in debug mode
        self.logger.debug(f"Request messages: {messages}")
        self.logger.debug(f"Request parameters: {merged_kwargs}")
        
        # Retry logic
        last_exception = None
        for attempt in range(self.default_retry_attempts):
            try:
                if attempt > 0:
                    self.logger.info(f"Retry attempt {attempt + 1}/{self.default_retry_attempts} for {model}")
                
                response = completion(
                    model=model,
                    messages=messages,
                    **merged_kwargs
                )
                
                # Track successful usage
                response_time = time.time() - start_time
                total_cost = litellm_get_total_cost(response)
                
                # Get detailed cost breakdown
                prompt_tokens = token_counter(model=model, messages=messages)
                completion_tokens = token_counter(model=model, messages=[response['choices'][0]['message']])
                total_tokens = prompt_tokens + completion_tokens
                
                # Calculate detailed costs
                prompt_cost, completion_cost, _= self.usage_tracker.calculate_token_costs(
                    model, prompt_tokens, completion_tokens
                )
                
                self.usage_tracker.track_usage(
                    model=model,
                    provider=provider,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    cost=total_cost,
                    request_id=request_id,
                    response_time=response_time,
                    success=True,
                    prompt_cost=prompt_cost,
                    completion_cost=completion_cost,
                )
                
                # Log successful response
                response_time_str = f"{response_time:.2f}s" if response_time is not None else "N/A"
                self.logger.info(f"LLM request successful: {model} - "
                               f"tokens: {total_tokens}, "
                               f"cost: ${total_cost:.4f}, "
                               f"response_time: {response_time_str}")
                
                # Log response content in debug mode
                if 'choices' in response and response['choices']:
                    response_content = response['choices'][0].get('message', {}).get('content', '')
                    if response_content:
                        self.logger.debug(f"Response content: {response_content[:200]}{'...' if len(response_content) > 200 else ''}")
                
                return response
                
            except (RateLimitError, APIConnectionError, APIError) as e:
                last_exception = e
                self.logger.warning(f"API error on attempt {attempt + 1}: {type(e).__name__}: {e}")
                
                if attempt < self.default_retry_attempts - 1:
                    retry_delay = self.default_retry_delay * (2 ** attempt)
                    self.logger.info(f"Retrying in {retry_delay:.1f}s... ({attempt + 1}/{self.default_retry_attempts})")
                    time.sleep(retry_delay)  # Exponential backoff
                    continue
                else:
                    # Track failed usage
                    response_time = time.time() - start_time
                    self.usage_tracker.track_usage(
                        model=model,
                        provider=provider,
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_tokens=0,
                        cost=0.0,
                        request_id=request_id,
                        response_time=response_time,
                        success=False,
                        error_message=str(e)
                    )
                    self.logger.error(f"LLM request failed after {self.default_retry_attempts} attempts: {type(e).__name__}: {e}")
                    raise e
            except Exception as e:
                # Track unexpected errors
                response_time = time.time() - start_time
                self.usage_tracker.track_usage(
                    model=model,
                    provider=provider,
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    cost=0.0,
                    request_id=request_id,
                    response_time=response_time,
                    success=False,
                    error_message=str(e)
                )
                self.logger.error(f"Unexpected error in LLM request: {type(e).__name__}: {e}")
                raise e
        
        # This should never be reached, but just in case
        raise last_exception or Exception("Unknown error occurred")
    
    def chat_completion(self,
                       messages: List[Dict[str, str]],
                       model: Optional[str] = None,
                       **kwargs) -> Dict[str, Any]:
        """
        Make a chat completion request.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            model: The model to use. If None, uses default_model.
            **kwargs: Additional arguments for the completion call
            
        Returns:
            Response dictionary from LiteLLM
        """
        return self._make_request(messages, model, **kwargs)
    
    def text_completion(self,
                       prompt: str,
                       model: Optional[str] = None,
                       **kwargs) -> Dict[str, Any]:
        """
        Make a text completion request.
        
        Args:
            prompt: The text prompt
            model: The model to use. If None, uses default_model.
            **kwargs: Additional arguments for the completion call
            
        Returns:
            Response dictionary from LiteLLM
        """
        messages = [{"role": "user", "content": prompt}]
        return self._make_request(messages, model, **kwargs)
    
    def get_usage_stats(self, **kwargs) -> UsageAggregate:
        """
        Get usage statistics.
        
        Args:
            **kwargs: Arguments for filtering (start_date, end_date, model_filter, provider_filter)
            
        Returns:
            UsageAggregate object with statistics
        """
        return self.usage_tracker.get_aggregated_usage(**kwargs)
    
    def save_usage_data(self, file_path: Optional[str] = None) -> None:
        """
        Save usage data to file.
        
        Args:
            file_path: Optional custom file path
        """
        self.usage_tracker.save_usage_data(file_path)
    
    def load_usage_data(self, file_path: Optional[str] = None) -> None:
        """
        Load usage data from file.
        
        Args:
            file_path: Optional custom file path
        """
        self.usage_tracker.load_usage_data(file_path)
    
    def export_usage_data(self, file_path: str, format: Literal['json', 'csv'] = 'json', checkpoint_name: Optional[str] = None) -> None:
        """
        Export usage data in different formats.
        
        Args:
            file_path: Path to save the exported data
            format: Export format ('json' or 'csv')
            checkpoint_name: Optional checkpoint name to export only data from that checkpoint
        """
        self.usage_tracker.export_usage_data(file_path, format, checkpoint_name)
    
    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """
        Count tokens in text using the appropriate tokenizer.
        
        Args:
            text: Text to count tokens for
            model: Model name. If None, uses default_model.
            
        Returns:
            Number of tokens
        """
        model = model or self.default_model
        if model is None:
            raise ValueError("No model provided and no default_model set")
        return self.usage_tracker.count_tokens(text, model)
    
    def count_message_tokens(self, messages: List[Dict[str, str]], model: Optional[str] = None) -> int:
        """
        Count tokens in messages using the appropriate tokenizer.
        
        Args:
            messages: List of message dictionaries
            model: Model name. If None, uses default_model.
            
        Returns:
            Number of tokens
        """
        model = model or self.default_model
        if model is None:
            raise ValueError("No model provided and no default_model set")
        return self.usage_tracker.count_message_tokens(messages, model)
    
    def estimate_cost(self, messages: List[Dict[str, str]], model: Optional[str] = None, 
                     estimated_completion_tokens: int = 100) -> Dict[str, Any]:
        """
        Estimate the cost of a request before making it.
        
        Args:
            messages: List of message dictionaries
            model: Model name. If None, uses default_model.
            estimated_completion_tokens: Estimated number of completion tokens
            
        Returns:
            Dictionary with cost estimates
        """
        model = model or self.default_model
        if model is None:
            raise ValueError("No model provided and no default_model set")
        
        prompt_tokens = self.count_message_tokens(messages, model)
        prompt_cost, completion_cost, total_cost = self.usage_tracker.calculate_token_costs(
            model, prompt_tokens, estimated_completion_tokens
        )
        
        return {
            "model": model,
            "prompt_tokens": prompt_tokens,
            "estimated_completion_tokens": estimated_completion_tokens,
            "estimated_total_tokens": prompt_tokens + estimated_completion_tokens,
            "prompt_cost": prompt_cost,
            "completion_cost": completion_cost,
            "total_cost": total_cost,
        }
    
    def get_model_info(self, model: Optional[str] = None) -> Dict[str, Any]:
        """
        Get comprehensive information about a model.
        
        Args:
            model: Model name. If None, uses default_model.
            
        Returns:
            Dictionary with model information
        """
        model = model or self.default_model
        if model is None:
            raise ValueError("No model provided and no default_model set")
        
        cost_info = self.usage_tracker.get_model_cost_info(model)
        
        return {
            "model": model,
            "cost_info": cost_info,
            "provider": get_llm_provider(model) if model else "unknown"
        }
    
    def register_custom_model(self, model_name: str, cost_dict: Dict[str, Any]) -> bool:
        """
        Register a custom model with cost information.
        
        Args:
            model_name: Name of the model to register
            cost_dict: Dictionary with cost information
            
        Returns:
            True if successful, False otherwise
        """
        return self.usage_tracker.register_custom_model(model_name, cost_dict)

    # Checkpoint APIs
    def start_usage_checkpoint(self, name: str) -> None:
        """
        Start a named usage checkpoint on the underlying tracker.
        """
        self.usage_tracker.start_usage_checkpoint(name)

    def end_usage_checkpoint(self, name: str) -> None:
        """
        End a named usage checkpoint on the underlying tracker.
        """
        self.usage_tracker.end_usage_checkpoint(name)

    def get_checkpoint_usage(self, name: str) -> Dict[str, Any]:
        """
        Get a human-friendly summary dict for a checkpoint, mirroring get_usage_summary.
        """
        stats = self.usage_tracker.get_checkpoint_usage(name)
        return {
            "total_requests": stats.total_requests,
            "total_tokens": stats.total_tokens,
            "total_cost": f"${stats.total_cost:.4f}",
            "average_response_time": f"{stats.average_response_time:.2f}s",
            "success_rate": f"{stats.success_rate:.1%}",
            "top_models": sorted(stats.model_breakdown.items(), key=lambda x: x[1]['requests'], reverse=True)[:5],
            "top_providers": sorted(stats.provider_breakdown.items(), key=lambda x: x[1]['requests'], reverse=True)[:5],
        }


# Convenience functions for quick usage
def create_llm_client(data_file: Optional[str] = None, 
                     default_model: Optional[str] = None,
                     default_temperature: Optional[float] = None,
                     default_max_tokens: Optional[int] = None,
                     default_api_base: Optional[str] = None,
                     logger: Optional[logging.Logger] = None) -> LLMClient:
    """
    Create a new LLM client with usage tracking.
    
    Args:
        data_file: Optional path for usage data persistence
        default_model: Default model to use for requests
        default_temperature: Default temperature for requests
        default_max_tokens: Default max_tokens for requests
        default_api_base: Default API base URL for requests
        logger: Optional logger instance
        
    Returns:
        LLMClient instance
    """
    tracker = LLMUsageTracker(data_file, logger)
    return LLMClient(
        tracker,
        default_model=default_model,
        default_temperature=default_temperature,
        default_max_tokens=default_max_tokens,
        default_api_base=default_api_base,
        logger=logger
    )


def get_usage_summary(client: LLMClient, **kwargs) -> Dict[str, Any]:
    """
    Get a summary of usage statistics.
    
    Args:
        client: LLMClient instance
        **kwargs: Filtering arguments
        
    Returns:
        Dictionary with usage summary
    """
    stats = client.get_usage_stats(**kwargs)
    return {
        "total_requests": stats.total_requests,
        "total_tokens": stats.total_tokens,
        "total_cost": f"${stats.total_cost:.4f}",
        "average_response_time": f"{stats.average_response_time:.2f}s",
        "success_rate": f"{stats.success_rate:.1%}",
        "top_models": sorted(stats.model_breakdown.items(), 
                           key=lambda x: x[1]['requests'], reverse=True)[:5],
        "top_providers": sorted(stats.provider_breakdown.items(), 
                              key=lambda x: x[1]['requests'], reverse=True)[:5]
    }


# Example usage and configuration
if __name__ == "__main__":
    # Set up logging
    setup_logging(level="INFO", log_file="llm_utils.log")
    
    # Example usage with default parameters
    client = LLMClient(
        default_model="gpt-3.5-turbo",
        default_temperature=0.7,
        default_max_tokens=1000,
        default_api_base=None  # Use default OpenAI API base
    )
    
    # Example: Token counting and cost estimation
    sample_text = "Hello, how are you today?"
    messages = [{"role": "user", "content": sample_text}]
    
    print("=== Token Counting and Cost Estimation ===")
    token_count = client.count_tokens(sample_text)
    message_token_count = client.count_message_tokens(messages)
    print(f"Text tokens: {token_count}")
    print(f"Message tokens: {message_token_count}")
    
    # Cost estimation
    cost_estimate = client.estimate_cost(messages, estimated_completion_tokens=50)
    print(f"Cost estimate: ${cost_estimate['total_cost']:.6f}")
    print(f"  - Prompt cost: ${cost_estimate['prompt_cost']:.6f}")
    print(f"  - Completion cost: ${cost_estimate['completion_cost']:.6f}")
    
    # Model information
    model_info = client.get_model_info()
    print(f"Model info: {model_info}")
    
    # Example: Encoding and decoding
    print("\n=== Encoding and Decoding ===")
    tokens = client.usage_tracker.encode_text(sample_text, "gpt-3.5-turbo")
    decoded_text = client.usage_tracker.decode_tokens(tokens, "gpt-3.5-turbo")
    print(f"Original: {sample_text}")
    print(f"Decoded: {decoded_text}")
    print(f"Tokens: {tokens[:10]}...")  # Show first 10 tokens
    
    # Example chat completion using default model and parameters
    print("\n=== Chat Completion ===")
    try:
        response = client.chat_completion(
            messages=[
                {"role": "user", "content": "Hello, how are you?"}
            ]
        )
        print("Response:", response['choices'][0]['message']['content'])
    except Exception as e:
        print(f"Error: {e}")
    
    # Example text completion with overridden temperature
    print("\n=== Text Completion ===")
    try:
        response = client.text_completion(
            prompt="Write a short poem about coding",
            temperature=0.9  # Override default temperature
        )
        print("Poem:", response['choices'][0]['message']['content'])
    except Exception as e:
        print(f"Error: {e}")
    
    # Get usage statistics
    print("\n=== Usage Statistics ===")
    stats = client.get_usage_stats()
    print(f"Total requests: {stats.total_requests}")
    print(f"Total cost: ${stats.total_cost:.4f}")
    print(f"Success rate: {stats.success_rate:.1%}")
    
    # Save usage data
    client.save_usage_data()
    print("Usage data saved!")

