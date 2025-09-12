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
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Literal
from dataclasses import dataclass, asdict
from collections import defaultdict

import litellm
from litellm import completion, completion_cost, get_llm_provider
from litellm.exceptions import RateLimitError, APIConnectionError, APIError


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
    
    def __init__(self, data_file: Optional[str] = None):
        """
        Initialize the usage tracker.
        
        Args:
            data_file: Path to the JSON file for persisting usage data.
                      If None, defaults to 'llm_usage_data.json' in current directory.
        """
        self.data_file = data_file or "llm_usage_data.json"
        self.usage_data: List[UsageData] = []
        self.lock = threading.Lock()
        
        # Load existing data if file exists
        self.load_usage_data()
    
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
                   error_message: Optional[str] = None) -> None:
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
            error_message=error_message
        )
        
        with self.lock:
            self.usage_data.append(usage)
    
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
    
    def save_usage_data(self, file_path: Optional[str] = None) -> None:
        """
        Save usage data to a JSON file.
        
        Args:
            file_path: Optional custom file path. If None, uses the default data_file.
        """
        save_path = file_path or self.data_file
        
        with self.lock:
            data_to_save = [asdict(usage) for usage in self.usage_data]
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(data_to_save, f, indent=2)
    
    def load_usage_data(self, file_path: Optional[str] = None) -> None:
        """
        Load usage data from a JSON file.
        
        Args:
            file_path: Optional custom file path. If None, uses the default data_file.
        """
        load_path = file_path or self.data_file
        
        if not os.path.exists(load_path):
            return
        
        try:
            with open(load_path, 'r') as f:
                data = json.load(f)
            
            with self.lock:
                self.usage_data = [UsageData(**item) for item in data]
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"Warning: Could not load usage data from {load_path}: {e}")
    
    def clear_usage_data(self) -> None:
        """Clear all usage data from memory."""
        with self.lock:
            self.usage_data.clear()
    
    def export_usage_data(self, file_path: str, format: Literal['json', 'csv'] = 'json') -> None:
        """
        Export usage data in different formats.
        
        Args:
            file_path: Path to save the exported data
            format: Export format ('json' or 'csv')
        """
        with self.lock:
            data_to_export = [asdict(usage) for usage in self.usage_data]
        
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


class LLMClient:
    """
    A comprehensive LLM client that supports multiple backends with usage tracking.
    
    This class provides a unified interface for making LLM calls across different
    providers while automatically tracking usage and costs.
    """
    
    def __init__(self, usage_tracker: Optional[LLMUsageTracker] = None, default_retry_attempts: int = 3, default_retry_delay: float = 1.0):
        """
        Initialize the LLM client.
        
        Args:
            usage_tracker: Optional usage tracker instance. If None, creates a new one.
            default_retry_attempts: Number of retry attempts for API errors
            default_retry_delay: Delay between retry attempts in seconds
        """
        self.usage_tracker = usage_tracker or LLMUsageTracker()
        self.default_retry_attempts = default_retry_attempts
        self.default_retry_delay = default_retry_delay
    
    def _make_request(self, 
                     model: str,
                     messages: List[Dict[str, str]],
                     **kwargs) -> Dict[str, Any]:
        """
        Make an LLM request with retry logic and usage tracking.
        
        Args:
            model: The model to use (e.g., 'gpt-4', 'claude-3-sonnet')
            messages: List of message dictionaries
            **kwargs: Additional arguments for the completion call
            
        Returns:
            Response dictionary from LiteLLM
        """
        start_time = time.time()
        request_id = kwargs.get('request_id')
        
        # Get provider information
        try:
            provider = get_llm_provider(model)
        except Exception:
            provider = "unknown"
        
        # Retry logic
        last_exception = None
        for attempt in range(self.default_retry_attempts):
            try:
                response = completion(
                    model=model,
                    messages=messages,
                    **kwargs
                )
                print(response)
                
                # Track successful usage
                response_time = time.time() - start_time
                usage_info = completion_cost(response)
                
                self.usage_tracker.track_usage(
                    model=model,
                    provider=provider,
                    prompt_tokens=usage_info.get('prompt_tokens', 0),
                    completion_tokens=usage_info.get('completion_tokens', 0),
                    total_tokens=usage_info.get('total_tokens', 0),
                    cost=usage_info.get('cost', 0.0),
                    request_id=request_id,
                    response_time=response_time,
                    success=True
                )
                
                return response
                
            except (RateLimitError, APIConnectionError, APIError) as e:
                last_exception = e
                if attempt < self.default_retry_attempts - 1:
                    print(f"Retrying... {attempt + 1}/{self.default_retry_attempts}")
                    time.sleep(self.default_retry_delay * (2 ** attempt))  # Exponential backoff
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
                raise e
        
        # This should never be reached, but just in case
        raise last_exception or Exception("Unknown error occurred")
    
    def chat_completion(self,
                       model: str,
                       messages: List[Dict[str, str]],
                       **kwargs) -> Dict[str, Any]:
        """
        Make a chat completion request.
        
        Args:
            model: The model to use
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional arguments for the completion call
            
        Returns:
            Response dictionary from LiteLLM
        """
        return self._make_request(model, messages, **kwargs)
    
    def text_completion(self,
                       model: str,
                       prompt: str,
                       **kwargs) -> Dict[str, Any]:
        """
        Make a text completion request.
        
        Args:
            model: The model to use
            prompt: The text prompt
            **kwargs: Additional arguments for the completion call
            
        Returns:
            Response dictionary from LiteLLM
        """
        messages = [{"role": "user", "content": prompt}]
        return self._make_request(model, messages, **kwargs)
    
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
    
    def export_usage_data(self, file_path: str, format: Literal['json', 'csv'] = 'json') -> None:
        """
        Export usage data in different formats.
        
        Args:
            file_path: Path to save the exported data
            format: Export format ('json' or 'csv')
        """
        self.usage_tracker.export_usage_data(file_path, format)


# Convenience functions for quick usage
def create_llm_client(data_file: Optional[str] = None) -> LLMClient:
    """
    Create a new LLM client with usage tracking.
    
    Args:
        data_file: Optional path for usage data persistence
        
    Returns:
        LLMClient instance
    """
    tracker = LLMUsageTracker(data_file)
    return LLMClient(tracker)


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
    # Example usage
    client = create_llm_client("example_usage.json")
    
    # Example chat completion
    try:
        response = client.chat_completion(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Hello, how are you?"}
            ]
        )
        print("Response:", response['choices'][0]['message']['content'])
    except Exception as e:
        print(f"Error: {e}")
    
    # Get usage statistics
    stats = client.get_usage_stats()
    print(f"Total requests: {stats.total_requests}")
    print(f"Total cost: ${stats.total_cost:.4f}")
    
    # Save usage data
    client.save_usage_data()

