"""
Comprehensive test suite for llm_utils module.

This module tests all classes and functionalities in llm_util.py including:
- UsageData and UsageAggregate dataclasses
- LLMUsageTracker class
- LLMClient class
- Convenience functions
- Edge cases and error handling
"""

import json
import logging
import os
import tempfile
import threading
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
import csv

# Import the module under test
# Add the parent directory to the Python path to find the llm_utils module
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_utils.llm_util import (
    UsageData, UsageAggregate, LLMUsageTracker, LLMClient,
    create_llm_client, get_usage_summary, setup_logging, get_logger
)


class TestUsageData(unittest.TestCase):
    """Test cases for UsageData dataclass."""
    
    def test_usage_data_creation(self):
        """Test basic UsageData creation with all fields."""
        usage = UsageData(
            timestamp="2024-01-01T00:00:00Z",
            model="gpt-4",
            provider="openai",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cost=0.03,
            request_id="req-123",
            response_time=1.5,
            success=True,
            error_message=None,
            prompt_cost=0.02,
            completion_cost=0.01
        )
        
        self.assertEqual(usage.timestamp, "2024-01-01T00:00:00Z")
        self.assertEqual(usage.model, "gpt-4")
        self.assertEqual(usage.provider, "openai")
        self.assertEqual(usage.prompt_tokens, 100)
        self.assertEqual(usage.completion_tokens, 50)
        self.assertEqual(usage.total_tokens, 150)
        self.assertEqual(usage.cost, 0.03)
        self.assertEqual(usage.request_id, "req-123")
        self.assertEqual(usage.response_time, 1.5)
        self.assertTrue(usage.success)
        self.assertIsNone(usage.error_message)
        self.assertEqual(usage.prompt_cost, 0.02)
        self.assertEqual(usage.completion_cost, 0.01)
    
    def test_usage_data_defaults(self):
        """Test UsageData creation with default values."""
        usage = UsageData(
            timestamp="2024-01-01T00:00:00Z",
            model="gpt-3.5-turbo",
            provider="openai",
            prompt_tokens=50,
            completion_tokens=25,
            total_tokens=75,
            cost=0.015
        )
        
        self.assertIsNone(usage.request_id)
        self.assertIsNone(usage.response_time)
        self.assertTrue(usage.success)
        self.assertIsNone(usage.error_message)
        self.assertIsNone(usage.prompt_cost)
        self.assertIsNone(usage.completion_cost)


class TestUsageAggregate(unittest.TestCase):
    """Test cases for UsageAggregate dataclass."""
    
    def test_usage_aggregate_creation(self):
        """Test basic UsageAggregate creation."""
        aggregate = UsageAggregate(
            total_requests=10,
            total_tokens=1000,
            total_cost=0.2,
            total_prompt_tokens=600,
            total_completion_tokens=400,
            average_response_time=1.5,
            success_rate=0.9,
            model_breakdown={"gpt-4": {"requests": 5, "tokens": 500}},
            provider_breakdown={"openai": {"requests": 10, "tokens": 1000}},
            time_range={"start": "2024-01-01", "end": "2024-01-02"}
        )
        
        self.assertEqual(aggregate.total_requests, 10)
        self.assertEqual(aggregate.total_tokens, 1000)
        self.assertEqual(aggregate.total_cost, 0.2)
        self.assertEqual(aggregate.total_prompt_tokens, 600)
        self.assertEqual(aggregate.total_completion_tokens, 400)
        self.assertEqual(aggregate.average_response_time, 1.5)
        self.assertEqual(aggregate.success_rate, 0.9)
        self.assertIn("gpt-4", aggregate.model_breakdown)
        self.assertIn("openai", aggregate.provider_breakdown)


class TestLLMUsageTracker(unittest.TestCase):
    """Test cases for LLMUsageTracker class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_file = os.path.join(self.temp_dir, "test_usage.json")
        self.tracker = LLMUsageTracker(self.temp_file)
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up temporary files
        if os.path.exists(self.temp_file):
            os.remove(self.temp_file)
        os.rmdir(self.temp_dir)
    
    def test_tracker_initialization(self):
        """Test tracker initialization with custom data file."""
        self.assertEqual(self.tracker.data_file, self.temp_file)
        self.assertEqual(len(self.tracker.usage_data), 0)
        self.assertIsInstance(self.tracker.lock, type(threading.Lock()))
    
    def test_tracker_initialization_default_file(self):
        """Test tracker initialization with default data file."""
        tracker = LLMUsageTracker()
        self.assertEqual(tracker.data_file, "llm_usage_data.json")
    
    def test_track_usage_success(self):
        """Test tracking successful usage."""
        self.tracker.track_usage(
            model="gpt-4",
            provider="openai",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cost=0.03,
            request_id="req-123",
            response_time=1.5
        )
        
        self.assertEqual(len(self.tracker.usage_data), 1)
        usage = self.tracker.usage_data[0]
        self.assertEqual(usage.model, "gpt-4")
        self.assertEqual(usage.provider, "openai")
        self.assertEqual(usage.prompt_tokens, 100)
        self.assertEqual(usage.completion_tokens, 50)
        self.assertEqual(usage.total_tokens, 150)
        self.assertEqual(usage.cost, 0.03)
        self.assertEqual(usage.request_id, "req-123")
        self.assertEqual(usage.response_time, 1.5)
        self.assertTrue(usage.success)
        self.assertIsNone(usage.error_message)
    
    def test_track_usage_failure(self):
        """Test tracking failed usage."""
        self.tracker.track_usage(
            model="gpt-4",
            provider="openai",
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            cost=0.0,
            success=False,
            error_message="Rate limit exceeded"
        )
        
        self.assertEqual(len(self.tracker.usage_data), 1)
        usage = self.tracker.usage_data[0]
        self.assertFalse(usage.success)
        self.assertEqual(usage.error_message, "Rate limit exceeded")
    
    def test_get_aggregated_usage_empty(self):
        """Test aggregated usage with no data."""
        aggregate = self.tracker.get_aggregated_usage()
        
        self.assertEqual(aggregate.total_requests, 0)
        self.assertEqual(aggregate.total_tokens, 0)
        self.assertEqual(aggregate.total_cost, 0.0)
        self.assertEqual(aggregate.success_rate, 0.0)
        self.assertEqual(aggregate.model_breakdown, {})
        self.assertEqual(aggregate.provider_breakdown, {})
    
    def test_get_aggregated_usage_with_data(self):
        """Test aggregated usage with sample data."""
        # Add some test data
        self.tracker.track_usage("gpt-4", "openai", 100, 50, 150, 0.03, response_time=1.0)
        self.tracker.track_usage("gpt-3.5-turbo", "openai", 80, 40, 120, 0.02, response_time=0.8)
        self.tracker.track_usage("claude-3", "anthropic", 90, 45, 135, 0.025, response_time=1.2)
        
        aggregate = self.tracker.get_aggregated_usage()
        
        self.assertEqual(aggregate.total_requests, 3)
        self.assertEqual(aggregate.total_tokens, 405)
        self.assertEqual(aggregate.total_cost, 0.075)
        self.assertEqual(aggregate.total_prompt_tokens, 270)
        self.assertEqual(aggregate.total_completion_tokens, 135)
        self.assertAlmostEqual(aggregate.average_response_time, 1.0, places=1)
        self.assertEqual(aggregate.success_rate, 1.0)
        
        # Check model breakdown
        self.assertIn("gpt-4", aggregate.model_breakdown)
        self.assertIn("gpt-3.5-turbo", aggregate.model_breakdown)
        self.assertIn("claude-3", aggregate.model_breakdown)
        
        # Check provider breakdown
        self.assertIn("openai", aggregate.provider_breakdown)
        self.assertIn("anthropic", aggregate.provider_breakdown)
    
    def test_get_aggregated_usage_with_filters(self):
        """Test aggregated usage with date and model filters."""
        # Add test data with different timestamps
        base_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        
        # Mock the track_usage method to set specific timestamps
        with patch.object(self.tracker, 'track_usage') as mock_track:
            # Create usage data manually to control timestamps
            usage1 = UsageData(
                timestamp=(base_time).isoformat(),
                model="gpt-4",
                provider="openai",
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
                cost=0.03
            )
            usage2 = UsageData(
                timestamp=(base_time.replace(day=2)).isoformat(),
                model="gpt-3.5-turbo",
                provider="openai",
                prompt_tokens=80,
                completion_tokens=40,
                total_tokens=120,
                cost=0.02
            )
            usage3 = UsageData(
                timestamp=(base_time.replace(day=3)).isoformat(),
                model="gpt-4",
                provider="openai",
                prompt_tokens=90,
                completion_tokens=45,
                total_tokens=135,
                cost=0.025
            )
            
            self.tracker.usage_data = [usage1, usage2, usage3]
        
        # Test date filtering
        start_date = datetime(2024, 1, 2, tzinfo=timezone.utc)
        end_date = datetime(2024, 1, 2, 23, 59, 59, tzinfo=timezone.utc)
        
        aggregate = self.tracker.get_aggregated_usage(
            start_date=start_date,
            end_date=end_date
        )
        self.assertEqual(aggregate.total_requests, 1)
        self.assertEqual(aggregate.model_breakdown["gpt-3.5-turbo"]["requests"], 1)
        
        # Test model filtering
        aggregate = self.tracker.get_aggregated_usage(model_filter="gpt-4")
        self.assertEqual(aggregate.total_requests, 2)
        self.assertIn("gpt-4", aggregate.model_breakdown)
        self.assertNotIn("gpt-3.5-turbo", aggregate.model_breakdown)
        
        # Test provider filtering
        aggregate = self.tracker.get_aggregated_usage(provider_filter="openai")
        self.assertEqual(aggregate.total_requests, 3)
    
    def test_save_and_load_usage_data(self):
        """Test saving and loading usage data."""
        # Add some test data
        self.tracker.track_usage("gpt-4", "openai", 100, 50, 150, 0.03)
        self.tracker.track_usage("gpt-3.5-turbo", "openai", 80, 40, 120, 0.02)
        
        # Save data
        self.tracker.save_usage_data()
        
        # Create new tracker and load data
        new_tracker = LLMUsageTracker(self.temp_file)
        self.assertEqual(len(new_tracker.usage_data), 2)
        self.assertEqual(new_tracker.usage_data[0].model, "gpt-4")
        self.assertEqual(new_tracker.usage_data[1].model, "gpt-3.5-turbo")
    
    def test_load_nonexistent_file(self):
        """Test loading from non-existent file."""
        nonexistent_file = os.path.join(self.temp_dir, "nonexistent.json")
        tracker = LLMUsageTracker(nonexistent_file)
        self.assertEqual(len(tracker.usage_data), 0)
    
    def test_clear_usage_data(self):
        """Test clearing usage data."""
        self.tracker.track_usage("gpt-4", "openai", 100, 50, 150, 0.03)
        self.assertEqual(len(self.tracker.usage_data), 1)
        
        self.tracker.clear_usage_data()
        self.assertEqual(len(self.tracker.usage_data), 0)
    
    def test_export_usage_data_json(self):
        """Test exporting usage data as JSON."""
        self.tracker.track_usage("gpt-4", "openai", 100, 50, 150, 0.03)
        
        export_file = os.path.join(self.temp_dir, "export.json")
        self.tracker.export_usage_data(export_file, format='json')
        
        self.assertTrue(os.path.exists(export_file))
        with open(export_file, 'r') as f:
            data = json.load(f)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]['model'], "gpt-4")

        # Clean up
        os.remove(export_file)
    
    def test_export_usage_data_csv(self):
        """Test exporting usage data as CSV."""
        self.tracker.track_usage("gpt-4", "openai", 100, 50, 150, 0.03)
        
        export_file = os.path.join(self.temp_dir, "export.csv")
        self.tracker.export_usage_data(export_file, format='csv')
        
        self.assertTrue(os.path.exists(export_file))
        with open(export_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['model'], "gpt-4")

        # Clean up
        os.remove(export_file)
    
    def test_thread_safety(self):
        """Test thread safety of usage tracking."""
        def track_usage_worker(tracker, model, count):
            for i in range(count):
                tracker.track_usage(
                    model=model,
                    provider="openai",
                    prompt_tokens=10,
                    completion_tokens=5,
                    total_tokens=15,
                    cost=0.001
                )
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(
                target=track_usage_worker,
                args=(self.tracker, f"model-{i}", 10)
            )
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check that all usage data was tracked
        self.assertEqual(len(self.tracker.usage_data), 50)


class TestLLMClient(unittest.TestCase):
    """Test cases for LLMClient class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_file = os.path.join(self.temp_dir, "test_client_usage.json")
        self.tracker = LLMUsageTracker(self.temp_file)
        self.client = LLMClient(self.tracker)
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_file):
            os.remove(self.temp_file)
        os.rmdir(self.temp_dir)
    
    def test_client_initialization(self):
        """Test client initialization."""
        self.assertEqual(self.client.usage_tracker, self.tracker)
        self.assertEqual(self.client.default_retry_attempts, 3)
        self.assertEqual(self.client.default_retry_delay, 1.0)
        self.assertIsNone(self.client.default_model)
        self.assertIsNone(self.client.default_temperature)
        self.assertIsNone(self.client.default_max_tokens)
        self.assertIsNone(self.client.default_api_base)
    
    def test_client_initialization_defaults(self):
        """Test client initialization with default tracker."""
        client = LLMClient()
        self.assertIsInstance(client.usage_tracker, LLMUsageTracker)
    
    def test_client_initialization_with_default_params(self):
        """Test client initialization with default completion parameters."""
        client = LLMClient(
            default_model="gpt-3.5-turbo",
            default_temperature=0.7,
            default_max_tokens=1000,
            default_api_base="https://api.openai.com/v1"
        )
        
        self.assertEqual(client.default_model, "gpt-3.5-turbo")
        self.assertEqual(client.default_temperature, 0.7)
        self.assertEqual(client.default_max_tokens, 1000)
        self.assertEqual(client.default_api_base, "https://api.openai.com/v1")
    
    @patch('llm_utils.llm_util.completion')
    @patch('llm_utils.llm_util.litellm_get_total_cost')
    @patch('llm_utils.llm_util.token_counter')
    @patch('llm_utils.llm_util.get_llm_provider')
    def test_chat_completion_success(self, mock_get_provider, mock_token_counter, mock_completion_cost, mock_completion):
        """Test successful chat completion."""
        # Mock responses
        mock_get_provider.return_value = "openai"
        mock_completion.return_value = {
            'choices': [{'message': {'content': 'Hello!'}}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}
        }
        mock_completion_cost.return_value = 0.001  # Now returns float
        mock_token_counter.side_effect = [10, 5]  # prompt_tokens, completion_tokens
        
        messages = [{"role": "user", "content": "Hello"}]
        response = self.client.chat_completion(messages, model="gpt-3.5-turbo")
        
        self.assertEqual(response['choices'][0]['message']['content'], 'Hello!')
        mock_completion.assert_called_once()
        
        # Check that usage was tracked
        self.assertEqual(len(self.tracker.usage_data), 1)
        usage = self.tracker.usage_data[0]
        self.assertEqual(usage.model, "gpt-3.5-turbo")
        self.assertEqual(usage.provider, "openai")
        self.assertTrue(usage.success)
    
    @patch('llm_utils.llm_util.text_completion')
    @patch('llm_utils.llm_util.litellm_get_total_cost')
    @patch('llm_utils.llm_util.token_counter')
    @patch('llm_utils.llm_util.get_llm_provider')
    def test_text_completion_success(self, mock_get_provider, mock_token_counter, mock_completion_cost, mock_completion):
        """Test successful text completion."""
        # Mock responses
        mock_get_provider.return_value = "openai"
        mock_completion.return_value = {
            'choices': [{'text': 'Hello world!'}],
            'usage': {'prompt_tokens': 5, 'completion_tokens': 3, 'total_tokens': 8}
        }
        mock_completion_cost.return_value = 0.0005  # Now returns float
        mock_token_counter.side_effect = [5, 3]  # prompt_tokens, completion_tokens
        
        response = self.client.text_completion("Hello", model="gpt-3.5-turbo")
        
        self.assertEqual(response['choices'][0]['text'], 'Hello world!')
        mock_completion.assert_called_once()
        
        # Check that usage was tracked
        self.assertEqual(len(self.tracker.usage_data), 1)
        usage = self.tracker.usage_data[0]
        self.assertEqual(usage.model, "gpt-3.5-turbo")
        self.assertTrue(usage.success)
    
    @patch('llm_utils.llm_util.embedding')
    @patch('llm_utils.llm_util.litellm_get_total_cost')
    @patch('llm_utils.llm_util.token_counter')
    @patch('llm_utils.llm_util.get_llm_provider')
    def test_embedding_success(self, mock_get_provider, mock_token_counter, mock_embedding_cost, mock_embedding):
        """Test successful embedding request."""
        # Mock responses
        mock_get_provider.return_value = "openai"
        mock_embedding.return_value = {
            'data': [{'embedding': [0.1, 0.2, 0.3]}],
            'usage': {'total_tokens': 5, 'prompt_tokens': 5}
        }
        mock_embedding_cost.return_value = 0.0001  # Now returns float
        mock_token_counter.side_effect = [5]  # Only prompt tokens for embeddings
        
        response = self.client.embedding(input="Hello world", model="text-embedding-ada-002")
        
        self.assertEqual(len(response['data']), 1)
        self.assertEqual(len(response['data'][0]['embedding']), 3)
        mock_embedding.assert_called_once()
        
        # Check that usage was tracked
        self.assertEqual(len(self.tracker.usage_data), 1)
        usage = self.tracker.usage_data[0]
        self.assertEqual(usage.model, "text-embedding-ada-002")
        self.assertEqual(usage.provider, "openai")
        self.assertEqual(usage.prompt_tokens, 5)
        self.assertEqual(usage.completion_tokens, 0)
        self.assertTrue(usage.success)
    
    @patch('llm_utils.llm_util.embedding')
    @patch('llm_utils.llm_util.litellm_get_total_cost')
    @patch('llm_utils.llm_util.token_counter')
    @patch('llm_utils.llm_util.get_llm_provider')
    def test_embedding_list_input(self, mock_get_provider, mock_token_counter, mock_embedding_cost, mock_embedding):
        """Test embedding with list of strings."""
        # Mock responses
        mock_get_provider.return_value = "openai"
        mock_embedding.return_value = {
            'data': [
                {'embedding': [0.1, 0.2]},
                {'embedding': [0.3, 0.4]}
            ],
            'usage': {'total_tokens': 10, 'prompt_tokens': 10}
        }
        mock_embedding_cost.return_value = 0.0002  # Now returns float
        mock_token_counter.side_effect = [5, 5]  # Two inputs
        
        response = self.client.embedding(input=["Hello", "World"], model="text-embedding-ada-002")
        
        self.assertEqual(len(response['data']), 2)
        mock_embedding.assert_called_once()
        
        # Check that usage was tracked
        self.assertEqual(len(self.tracker.usage_data), 1)
        usage = self.tracker.usage_data[0]
        self.assertEqual(usage.model, "text-embedding-ada-002")
        self.assertEqual(usage.prompt_tokens, 10)
        self.assertEqual(usage.completion_tokens, 0)
    
    @patch('llm_utils.llm_util.embedding')
    @patch('llm_utils.llm_util.litellm_get_total_cost')
    @patch('llm_utils.llm_util.token_counter')
    @patch('llm_utils.llm_util.get_llm_provider')
    def test_embedding_no_usage_in_response(self, mock_get_provider, mock_token_counter, mock_embedding_cost, mock_embedding):
        """Test embedding when response doesn't have usage info - should fall back to token counting."""
        # Mock responses
        mock_get_provider.return_value = "openai"
        mock_embedding.return_value = {
            'data': [{'embedding': [0.1, 0.2, 0.3]}]
            # No usage field
        }
        mock_embedding_cost.return_value = 0.0001  # Now returns float
        mock_token_counter.side_effect = [5]  # Fallback token counting
        
        response = self.client.embedding(input="Hello world", model="text-embedding-ada-002")
        
        self.assertEqual(len(response['data']), 1)
        mock_embedding.assert_called_once()
        
        # Check that usage was tracked (using fallback token counting)
        self.assertEqual(len(self.tracker.usage_data), 1)
        usage = self.tracker.usage_data[0]
        self.assertEqual(usage.model, "text-embedding-ada-002")
        self.assertEqual(usage.completion_tokens, 0)  # Embeddings always have 0 completion tokens
    
    @patch('llm_utils.llm_util.completion')
    @patch('llm_utils.llm_util.litellm_get_total_cost')
    @patch('llm_utils.llm_util.token_counter')
    @patch('llm_utils.llm_util.get_llm_provider')
    def test_retry_logic_on_rate_limit(self, mock_get_provider, mock_token_counter, mock_completion_cost, mock_completion):
        """Test retry logic on rate limit error."""
        # Import here to avoid import issues in test environment
        try:
            from litellm.exceptions import RateLimitError  # type: ignore
        except ImportError:
            # Create a mock exception if litellm is not available
            class RateLimitError(Exception):
                pass
        
        mock_get_provider.return_value = "openai"
        
        # First call fails with rate limit, second succeeds
        mock_completion.side_effect = [
            RateLimitError(500, "Rate limit exceeded", "openai", "gpt-3.5-turbo"),
            {'choices': [{'message': {'content': 'Success!'}}]}
        ]
        
        mock_completion_cost.return_value = 0.001  # Now returns float
        mock_token_counter.side_effect = [10, 5]  # prompt_tokens, completion_tokens
        
        messages = [{"role": "user", "content": "Hello"}]
        response = self.client.chat_completion(messages, model="gpt-3.5-turbo")

        self.assertEqual(response['choices'][0]['message']['content'], 'Success!')
        self.assertEqual(mock_completion.call_count, 2)
    
    @patch('llm_utils.llm_util.completion')
    @patch('llm_utils.llm_util.litellm_get_total_cost')
    @patch('llm_utils.llm_util.get_llm_provider')
    def test_track_failed_usage(self, mock_get_provider, mock_completion_cost, mock_completion):
        """Test tracking failed usage."""
        # Import here to avoid import issues in test environment
        try:
            from litellm.exceptions import APIError  # type: ignore
        except ImportError:
            # Create a mock exception if litellm is not available
            class APIError(Exception):
                pass
        
        mock_get_provider.return_value = "openai"
        mock_completion.side_effect = APIError(500, "API error", "openai", "gpt-3.5-turbo")
        
        messages = [{"role": "user", "content": "Hello"}]
        
        with self.assertRaises(APIError):
            self.client.chat_completion(messages, model="gpt-3.5-turbo")
        
        # Check that failed usage was tracked
        self.assertEqual(len(self.tracker.usage_data), 1)
        usage = self.tracker.usage_data[0]
        self.assertFalse(usage.success)
        self.assertIn("API error", usage.error_message)
    
    @patch('llm_utils.llm_util.embedding')
    @patch('llm_utils.llm_util.litellm_get_total_cost')
    @patch('llm_utils.llm_util.get_llm_provider')
    def test_embedding_track_failed_usage(self, mock_get_provider, mock_embedding_cost, mock_embedding):
        """Test tracking failed embedding usage."""
        # Import here to avoid import issues in test environment
        try:
            from litellm.exceptions import APIError  # type: ignore
        except ImportError:
            # Create a mock exception if litellm is not available
            class APIError(Exception):
                pass
        
        mock_get_provider.return_value = "openai"
        mock_embedding.side_effect = APIError(500, "API error", "openai", "text-embedding-ada-002")
        
        with self.assertRaises(APIError):
            self.client.embedding(input="Hello", model="text-embedding-ada-002")
        
        # Check that failed usage was tracked
        self.assertEqual(len(self.tracker.usage_data), 1)
        usage = self.tracker.usage_data[0]
        self.assertFalse(usage.success)
        self.assertIn("API error", usage.error_message)
        self.assertEqual(usage.model, "text-embedding-ada-002")
    
    def test_get_usage_stats(self):
        """Test getting usage statistics."""
        # Add some test data
        self.tracker.track_usage("gpt-4", "openai", 100, 50, 150, 0.03)
        
        stats = self.client.get_usage_stats()
        self.assertEqual(stats.total_requests, 1)
        self.assertEqual(stats.total_tokens, 150)
    
    def test_save_and_load_usage_data(self):
        """Test saving and loading usage data through client."""
        # Add test data
        self.tracker.track_usage("gpt-4", "openai", 100, 50, 150, 0.03)
        
        # Save data
        self.client.save_usage_data()
        
        # Create new client and load data
        new_tracker = LLMUsageTracker(self.temp_file)
        new_client = LLMClient(new_tracker)
        new_client.load_usage_data()
        
        self.assertEqual(len(new_tracker.usage_data), 1)
    
    def test_export_usage_data(self):
        """Test exporting usage data through client."""
        # Add test data
        self.tracker.track_usage("gpt-4", "openai", 100, 50, 150, 0.03)
        
        export_file = os.path.join(self.temp_dir, "client_export.json")
        self.client.export_usage_data(export_file, format='json')
        
        self.assertTrue(os.path.exists(export_file))

        # Clean up
        os.remove(export_file)


class TestDefaultParameters(unittest.TestCase):
    """Test cases for default parameter functionality in LLMClient."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_file = os.path.join(self.temp_dir, "default_params_test.json")
        self.tracker = LLMUsageTracker(self.temp_file)
        self.client = LLMClient(
            self.tracker,
            default_model="gpt-3.5-turbo",
            default_temperature=0.7,
            default_max_tokens=1000,
            default_api_base="https://api.openai.com/v1"
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_file):
            os.remove(self.temp_file)
        os.rmdir(self.temp_dir)
    
    @patch('llm_utils.llm_util.completion')
    @patch('llm_utils.llm_util.litellm_get_total_cost')
    @patch('llm_utils.llm_util.token_counter')
    @patch('llm_utils.llm_util.get_llm_provider')
    def test_chat_completion_with_defaults(self, mock_get_provider, mock_token_counter, mock_completion_cost, mock_completion):
        """Test chat completion using all default parameters."""
        mock_get_provider.return_value = "openai"
        mock_completion.return_value = {
            'choices': [{'message': {'content': 'Hello!'}}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}
        }
        mock_completion_cost.return_value = 0.001  # Now returns float
        mock_token_counter.side_effect = [10, 5]  # prompt_tokens, completion_tokens
        
        messages = [{"role": "user", "content": "Hello"}]
        response = self.client.chat_completion(messages)
        
        self.assertEqual(response['choices'][0]['message']['content'], 'Hello!')
        
        # Check that completion was called with default parameters
        mock_completion.assert_called_once()
        call_args = mock_completion.call_args
        self.assertEqual(call_args[1]['model'], "gpt-3.5-turbo")
        self.assertEqual(call_args[1]['temperature'], 0.7)
        self.assertEqual(call_args[1]['max_tokens'], 1000)
        self.assertEqual(call_args[1]['api_base'], "https://api.openai.com/v1")
    
    @patch('llm_utils.llm_util.text_completion')
    @patch('llm_utils.llm_util.litellm_get_total_cost')
    @patch('llm_utils.llm_util.token_counter')
    @patch('llm_utils.llm_util.get_llm_provider')
    def test_text_completion_with_defaults(self, mock_get_provider, mock_token_counter, mock_completion_cost, mock_completion):
        """Test text completion using all default parameters."""
        mock_get_provider.return_value = "openai"
        mock_completion.return_value = {
            'choices': [{'text': 'Hello world!'}],
            'usage': {'prompt_tokens': 5, 'completion_tokens': 3, 'total_tokens': 8}
        }
        mock_completion_cost.return_value = 0.0005  # Now returns float
        mock_token_counter.side_effect = [5, 3]  # prompt_tokens, completion_tokens
        
        response = self.client.text_completion("Hello")
        
        self.assertEqual(response['choices'][0]['text'], 'Hello world!')
        
        # Check that completion was called with default parameters
        mock_completion.assert_called_once()
        call_args = mock_completion.call_args
        self.assertEqual(call_args[1]['model'], "gpt-3.5-turbo")
        self.assertEqual(call_args[1]['temperature'], 0.7)
        self.assertEqual(call_args[1]['max_tokens'], 1000)
        self.assertEqual(call_args[1]['api_base'], "https://api.openai.com/v1")
    
    @patch('llm_utils.llm_util.embedding')
    @patch('llm_utils.llm_util.litellm_get_total_cost')
    @patch('llm_utils.llm_util.token_counter')
    @patch('llm_utils.llm_util.get_llm_provider')
    def test_embedding_with_defaults(self, mock_get_provider, mock_token_counter, mock_embedding_cost, mock_embedding):
        """Test embedding using all default parameters."""
        mock_get_provider.return_value = "openai"
        mock_embedding.return_value = {
            'data': [{'embedding': [0.1, 0.2, 0.3]}],
            'usage': {'total_tokens': 5, 'prompt_tokens': 5}
        }
        mock_embedding_cost.return_value = 0.0001  # Now returns float
        mock_token_counter.side_effect = [5]  # prompt_tokens
        
        # Create client with default embedding model
        client_with_embedding_model = LLMClient(
            self.tracker,
            default_model="text-embedding-ada-002"
        )
        
        response = client_with_embedding_model.embedding(input="Hello")
        
        self.assertEqual(len(response['data']), 1)
        
        # Check that embedding was called with default parameters
        mock_embedding.assert_called_once()
        call_args = mock_embedding.call_args
        self.assertEqual(call_args[1]['model'], "text-embedding-ada-002")
    
    @patch('llm_utils.llm_util.completion')
    @patch('llm_utils.llm_util.litellm_get_total_cost')
    @patch('llm_utils.llm_util.token_counter')
    @patch('llm_utils.llm_util.get_llm_provider')
    def test_override_default_temperature(self, mock_get_provider, mock_token_counter, mock_completion_cost, mock_completion):
        """Test overriding default temperature parameter."""
        mock_get_provider.return_value = "openai"
        mock_completion.return_value = {
            'choices': [{'message': {'content': 'Hello!'}}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}
        }
        mock_completion_cost.return_value = 0.001  # Now returns float
        mock_token_counter.side_effect = [10, 5]  # prompt_tokens, completion_tokens
        
        messages = [{"role": "user", "content": "Hello"}]
        response = self.client.chat_completion(messages, temperature=0.9)
        
        # Check that completion was called with overridden temperature
        mock_completion.assert_called_once()
        call_args = mock_completion.call_args
        self.assertEqual(call_args[1]['temperature'], 0.9)
        self.assertEqual(call_args[1]['model'], "gpt-3.5-turbo")  # Still uses default model
        self.assertEqual(call_args[1]['max_tokens'], 1000)  # Still uses default max_tokens
    
    @patch('llm_utils.llm_util.completion')
    @patch('llm_utils.llm_util.litellm_get_total_cost')
    @patch('llm_utils.llm_util.token_counter')
    @patch('llm_utils.llm_util.get_llm_provider')
    def test_override_default_model(self, mock_get_provider, mock_token_counter, mock_completion_cost, mock_completion):
        """Test overriding default model parameter."""
        mock_get_provider.return_value = "openai"
        mock_completion.return_value = {
            'choices': [{'message': {'content': 'Hello!'}}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}
        }
        mock_completion_cost.return_value = 0.001  # Now returns float
        mock_token_counter.side_effect = [10, 5]  # prompt_tokens, completion_tokens
        
        messages = [{"role": "user", "content": "Hello"}]
        response = self.client.chat_completion(messages, model="gpt-4")
        
        # Check that completion was called with overridden model
        mock_completion.assert_called_once()
        call_args = mock_completion.call_args
        self.assertEqual(call_args[1]['model'], "gpt-4")
        self.assertEqual(call_args[1]['temperature'], 0.7)  # Still uses default temperature
        self.assertEqual(call_args[1]['max_tokens'], 1000)  # Still uses default max_tokens
    
    @patch('llm_utils.llm_util.completion')
    @patch('llm_utils.llm_util.litellm_get_total_cost')
    @patch('llm_utils.llm_util.token_counter')
    @patch('llm_utils.llm_util.get_llm_provider')
    def test_override_multiple_defaults(self, mock_get_provider, mock_token_counter, mock_completion_cost, mock_completion):
        """Test overriding multiple default parameters."""
        mock_get_provider.return_value = "openai"
        mock_completion.return_value = {
            'choices': [{'message': {'content': 'Hello!'}}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}
        }
        mock_completion_cost.return_value = 0.001  # Now returns float
        mock_token_counter.side_effect = [10, 5]  # prompt_tokens, completion_tokens
        
        messages = [{"role": "user", "content": "Hello"}]
        response = self.client.chat_completion(
            messages, 
            model="gpt-4",
            temperature=0.9,
            max_tokens=500
        )
        
        # Check that completion was called with overridden parameters
        mock_completion.assert_called_once()
        call_args = mock_completion.call_args
        self.assertEqual(call_args[1]['model'], "gpt-4")
        self.assertEqual(call_args[1]['temperature'], 0.9)
        self.assertEqual(call_args[1]['max_tokens'], 500)
        self.assertEqual(call_args[1]['api_base'], "https://api.openai.com/v1")  # Still uses default api_base
    
    def test_no_default_model_error(self):
        """Test error when no model is provided and no default is set."""
        client_no_default = LLMClient(self.tracker)  # No default model
        
        messages = [{"role": "user", "content": "Hello"}]
        
        with self.assertRaises(ValueError) as context:
            client_no_default.chat_completion(messages)
        
        self.assertIn("No model provided and no default_model set", str(context.exception))
    
    @patch('llm_utils.llm_util.completion')
    @patch('llm_utils.llm_util.litellm_get_total_cost')
    @patch('llm_utils.llm_util.token_counter')
    @patch('llm_utils.llm_util.get_llm_provider')
    def test_partial_defaults(self, mock_get_provider, mock_token_counter, mock_completion_cost, mock_completion):
        """Test client with only some default parameters set."""
        client_partial = LLMClient(
            self.tracker,
            default_model="gpt-3.5-turbo",
            default_temperature=0.8
            # No default max_tokens or api_base
        )
        
        mock_get_provider.return_value = "openai"
        mock_completion.return_value = {
            'choices': [{'message': {'content': 'Hello!'}}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}
        }
        mock_completion_cost.return_value = 0.001  # Now returns float
        mock_token_counter.side_effect = [10, 5]  # prompt_tokens, completion_tokens
        
        messages = [{"role": "user", "content": "Hello"}]
        response = client_partial.chat_completion(messages)
        
        # Check that completion was called with only the set defaults
        mock_completion.assert_called_once()
        call_args = mock_completion.call_args
        self.assertEqual(call_args[1]['model'], "gpt-3.5-turbo")
        self.assertEqual(call_args[1]['temperature'], 0.8)
        self.assertNotIn('max_tokens', call_args[1])
        self.assertNotIn('api_base', call_args[1])


class TestRefactoredSharedMethods(unittest.TestCase):
    """Test cases for the refactored shared methods in LLMClient."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_file = os.path.join(self.temp_dir, "refactored_test.json")
        self.tracker = LLMUsageTracker(self.temp_file)
        self.client = LLMClient(
            self.tracker,
            default_model="gpt-3.5-turbo",
            default_temperature=0.7,
            default_max_tokens=1000,
            default_api_base="https://api.openai.com/v1"
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_file):
            os.remove(self.temp_file)
        os.rmdir(self.temp_dir)
    
    def test_prepare_request_without_defaults(self):
        """Test _prepare_request method without default parameters."""
        client_no_defaults = LLMClient(self.tracker)
        
        with self.assertRaises(ValueError):
            client_no_defaults._prepare_request(None)
    
    def test_prepare_request_with_model_override(self):
        """Test _prepare_request method with model override."""
        model, provider, merged_kwargs = self.client._prepare_request("gpt-4", temperature=0.5)
        
        self.assertEqual(model, "gpt-4")
        self.assertEqual(merged_kwargs['temperature'], 0.5)
        self.assertEqual(merged_kwargs['max_tokens'], 1000)  # Still uses default
    
    def test_log_response_content_chat_completion(self):
        """Test _log_response_content method for chat completion."""
        response = {
            'choices': [{'message': {'content': 'Hello from chat completion!'}}]
        }
        
        # This should not raise an exception
        self.client._log_response_content(response, "chat completion")
    
    def test_log_response_content_text_completion(self):
        """Test _log_response_content method for text completion."""
        response = {
            'choices': [{'text': 'Hello from text completion!'}]
        }
        
        # This should not raise an exception
        self.client._log_response_content(response, "text completion")
    
    def test_log_response_content_empty_choices(self):
        """Test _log_response_content method with empty choices."""
        response = {'choices': []}
        
        # This should not raise an exception
        self.client._log_response_content(response, "test")
    
    def test_log_response_content_no_choices(self):
        """Test _log_response_content method with no choices key."""
        response = {}
        
        # This should not raise an exception
        self.client._log_response_content(response, "test")
    
    def test_log_response_content_embedding(self):
        """Test _log_response_content method for embedding."""
        response = {
            'data': [
                {'embedding': [0.1, 0.2, 0.3]},
                {'embedding': [0.4, 0.5, 0.6]}
            ]
        }
        
        # This should not raise an exception
        self.client._log_response_content(response, "embedding")
    
    def test_log_response_content_embedding_empty_data(self):
        """Test _log_response_content method with empty embedding data."""
        response = {'data': []}
        
        # This should not raise an exception
        self.client._log_response_content(response, "embedding")
    
    @patch('llm_utils.llm_util.completion')
    @patch('llm_utils.llm_util.litellm_get_total_cost')
    @patch('llm_utils.llm_util.token_counter')
    @patch('llm_utils.llm_util.get_llm_provider')
    def test_shared_execute_request_with_retry_chat(self, mock_get_provider, mock_token_counter, mock_completion_cost, mock_completion):
        """Test _execute_request_with_retry method with chat completion."""
        mock_get_provider.return_value = "openai"
        mock_completion.return_value = {
            'choices': [{'message': {'content': 'Hello!'}}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}
        }
        mock_completion_cost.return_value = 0.001
        mock_token_counter.side_effect = [10, 5]
        
        def request_func():
            from llm_utils.llm_util import completion
            return completion(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hello"}])
        
        def calculate_tokens(model, response, merged_kwargs):
            return 10, 5, 15
        
        # Temporarily override the token calculation method
        original_calculate = self.client._calculate_tokens_for_response
        self.client._calculate_tokens_for_response = calculate_tokens
        
        try:
            response = self.client._execute_request_with_retry(
                "gpt-3.5-turbo", "openai", {}, request_func, "chat completion"
            )
            
            self.assertEqual(response['choices'][0]['message']['content'], 'Hello!')
            self.assertEqual(len(self.tracker.usage_data), 1)
        finally:
            # Restore original method
            self.client._calculate_tokens_for_response = original_calculate
    
    @patch('llm_utils.llm_util.text_completion')
    @patch('llm_utils.llm_util.litellm_get_total_cost')
    @patch('llm_utils.llm_util.token_counter')
    @patch('llm_utils.llm_util.get_llm_provider')
    def test_shared_execute_request_with_retry_text(self, mock_get_provider, mock_token_counter, mock_completion_cost, mock_completion):
        """Test _execute_request_with_retry method with text completion."""
        mock_get_provider.return_value = "openai"
        mock_completion.return_value = {
            'choices': [{'text': 'Hello world!'}],
            'usage': {'prompt_tokens': 5, 'completion_tokens': 3, 'total_tokens': 8}
        }
        mock_completion_cost.return_value = 0.0005
        mock_token_counter.side_effect = [5, 3]
        
        def request_func():
            from llm_utils.llm_util import text_completion
            return text_completion(model="gpt-3.5-turbo", prompt="Hello")
        
        def calculate_tokens(model, response, merged_kwargs):
            return 5, 3, 8
        
        # Temporarily override the token calculation method
        original_calculate = self.client._calculate_tokens_for_response
        self.client._calculate_tokens_for_response = calculate_tokens
        
        try:
            response = self.client._execute_request_with_retry(
                "gpt-3.5-turbo", "openai", {}, request_func, "text completion"
            )
            
            self.assertEqual(response['choices'][0]['text'], 'Hello world!')
            self.assertEqual(len(self.tracker.usage_data), 1)
        finally:
            # Restore original method
            self.client._calculate_tokens_for_response = original_calculate
    
    @patch('llm_utils.llm_util.embedding')
    @patch('llm_utils.llm_util.litellm_get_total_cost')
    @patch('llm_utils.llm_util.token_counter')
    @patch('llm_utils.llm_util.get_llm_provider')
    def test_shared_execute_request_with_retry_embedding(self, mock_get_provider, mock_token_counter, mock_embedding_cost, mock_embedding):
        """Test _execute_request_with_retry method with embedding."""
        mock_get_provider.return_value = "openai"
        mock_embedding.return_value = {
            'data': [{'embedding': [0.1, 0.2, 0.3]}],
            'usage': {'total_tokens': 5, 'prompt_tokens': 5}
        }
        mock_embedding_cost.return_value = 0.0001
        mock_token_counter.side_effect = [5]
        
        def request_func():
            from llm_utils.llm_util import embedding
            return embedding(model="text-embedding-ada-002", input="Hello")
        
        def calculate_tokens(model, response, merged_kwargs):
            return 5, 0, 5  # prompt_tokens, completion_tokens, total_tokens
        
        # Temporarily override the token calculation method
        original_calculate = self.client._calculate_tokens_for_response
        self.client._calculate_tokens_for_response = calculate_tokens
        
        try:
            response = self.client._execute_request_with_retry(
                "text-embedding-ada-002", "openai", {}, request_func, "embedding"
            )
            
            self.assertEqual(len(response['data']), 1)
            self.assertEqual(len(self.tracker.usage_data), 1)
        finally:
            # Restore original method
            self.client._calculate_tokens_for_response = original_calculate


class TestConvenienceFunctions(unittest.TestCase):
    """Test cases for convenience functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_file = os.path.join(self.temp_dir, "convenience_test.json")
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_file):
            os.remove(self.temp_file)
        os.rmdir(self.temp_dir)
    
    def test_create_llm_client(self):
        """Test create_llm_client function."""
        client = create_llm_client(self.temp_file)
        
        self.assertIsInstance(client, LLMClient)
        self.assertIsInstance(client.usage_tracker, LLMUsageTracker)
        self.assertEqual(client.usage_tracker.data_file, self.temp_file)
        # Check that default parameters are None by default
        self.assertIsNone(client.default_model)
        self.assertIsNone(client.default_temperature)
        self.assertIsNone(client.default_max_tokens)
        self.assertIsNone(client.default_api_base)
    
    def test_create_llm_client_default(self):
        """Test create_llm_client function with default file."""
        client = create_llm_client()
        
        self.assertIsInstance(client, LLMClient)
        self.assertEqual(client.usage_tracker.data_file, "llm_usage_data.json")
    
    def test_create_llm_client_with_default_params(self):
        """Test create_llm_client function with default parameters."""
        client = create_llm_client(
            data_file=self.temp_file,
            default_model="gpt-3.5-turbo",
            default_temperature=0.7,
            default_max_tokens=1000,
            default_api_base="https://api.openai.com/v1"
        )
        
        self.assertIsInstance(client, LLMClient)
        self.assertEqual(client.usage_tracker.data_file, self.temp_file)
        self.assertEqual(client.default_model, "gpt-3.5-turbo")
        self.assertEqual(client.default_temperature, 0.7)
        self.assertEqual(client.default_max_tokens, 1000)
        self.assertEqual(client.default_api_base, "https://api.openai.com/v1")
    
    def test_get_usage_summary(self):
        """Test get_usage_summary function."""
        client = create_llm_client(self.temp_file)
        
        # Add some test data
        client.usage_tracker.track_usage("gpt-4", "openai", 100, 50, 150, 0.03, response_time=1.0)
        client.usage_tracker.track_usage("gpt-3.5-turbo", "openai", 80, 40, 120, 0.02, response_time=0.8)
        
        summary = get_usage_summary(client)
        
        self.assertEqual(summary["total_requests"], 2)
        self.assertEqual(summary["total_tokens"], 270)
        self.assertEqual(summary["total_cost"], "$0.0500")
        self.assertEqual(summary["average_response_time"], "0.90s")
        self.assertEqual(summary["success_rate"], "100.0%")
        self.assertIn("top_models", summary)
        self.assertIn("top_providers", summary)


class TestEdgeCasesAndErrorHandling(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_file = os.path.join(self.temp_dir, "edge_cases_test.json")
        self.tracker = LLMUsageTracker(self.temp_file)
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_file):
            os.remove(self.temp_file)
        os.rmdir(self.temp_dir)
    
    def test_aggregate_with_mixed_success_failure(self):
        """Test aggregation with mixed success and failure rates."""
        # Add successful usage
        self.tracker.track_usage("gpt-4", "openai", 100, 50, 150, 0.03, success=True)
        self.tracker.track_usage("gpt-4", "openai", 80, 40, 120, 0.02, success=True)
        
        # Add failed usage
        self.tracker.track_usage("gpt-4", "openai", 0, 0, 0, 0.0, success=False, error_message="Rate limit")
        
        aggregate = self.tracker.get_aggregated_usage()
        
        self.assertEqual(aggregate.total_requests, 3)
        self.assertEqual(aggregate.success_rate, 2/3)
        self.assertEqual(aggregate.total_tokens, 270)  # Only successful requests count
    
    def test_aggregate_with_no_response_times(self):
        """Test aggregation when no response times are available."""
        self.tracker.track_usage("gpt-4", "openai", 100, 50, 150, 0.03, response_time=None)
        self.tracker.track_usage("gpt-4", "openai", 80, 40, 120, 0.02, response_time=None)
        
        aggregate = self.tracker.get_aggregated_usage()
        
        self.assertEqual(aggregate.average_response_time, 0.0)
    
    def test_load_corrupted_json_file(self):
        """Test loading corrupted JSON file."""
        # Create a corrupted JSON file
        with open(self.temp_file, 'w') as f:
            f.write("invalid json content")
        
        # Should not raise exception, just print warning
        tracker = LLMUsageTracker(self.temp_file)
        self.assertEqual(len(tracker.usage_data), 0)
    
    def test_export_to_nonexistent_directory(self):
        """Test exporting to a directory that doesn't exist."""
        nonexistent_dir = os.path.join(self.temp_dir, "nonexistent", "export.json")
        
        self.tracker.track_usage("gpt-4", "openai", 100, 50, 150, 0.03)
        
        # Should create the directory
        self.tracker.export_usage_data(nonexistent_dir, format='json')
        self.assertTrue(os.path.exists(nonexistent_dir))

        # Clean up
        os.remove(nonexistent_dir)
        os.rmdir(os.path.dirname(nonexistent_dir))
    
    def test_aggregate_with_empty_model_breakdown(self):
        """Test aggregation with empty model breakdown."""
        aggregate = self.tracker.get_aggregated_usage()
        
        self.assertEqual(aggregate.model_breakdown, {})
        self.assertEqual(aggregate.provider_breakdown, {})
        self.assertEqual(aggregate.time_range, {"start": "", "end": ""})


class TestLogging(unittest.TestCase):
    """Test cases for logging functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_file = os.path.join(self.temp_dir, "test_logging.log")
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up temporary files
        for file in os.listdir(self.temp_dir):
            file_path = os.path.join(self.temp_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        os.rmdir(self.temp_dir)
    
    def test_setup_logging(self):
        """Test basic logging setup."""
        logger = setup_logging(level="INFO", log_file=self.temp_file)
        
        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.name, "llm_utils")
        self.assertEqual(logger.level, logging.INFO)
        
        # Test that log file is created
        logger.info("Test log message")
        self.assertTrue(os.path.exists(self.temp_file))
        
        # Check log content
        with open(self.temp_file, 'r') as f:
            log_content = f.read()
        self.assertIn("Test log message", log_content)
    
    def test_get_logger(self):
        """Test getting logger instances."""
        logger1 = get_logger("test_logger")
        logger2 = get_logger("test_logger")
        
        # Should return the same logger instance
        self.assertIs(logger1, logger2)
        self.assertEqual(logger1.name, "test_logger")
    
    def test_usage_tracker_logging(self):
        """Test logging in usage tracker."""
        # Set up logging
        setup_logging(level="INFO", log_file=self.temp_file)
        
        # Create tracker
        tracker = LLMUsageTracker()
        
        # Track usage - should log
        tracker.track_usage(
            model="gpt-3.5-turbo",
            provider="openai",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cost=0.03,
            response_time=1.5
        )
        
        # Check that log was written
        with open(self.temp_file, 'r') as f:
            log_content = f.read()
        self.assertIn("Tracked successful usage", log_content)
        self.assertIn("gpt-3.5-turbo", log_content)
        self.assertIn("openai", log_content)
    
    def test_usage_tracker_error_logging(self):
        """Test error logging in usage tracker."""
        # Set up logging
        setup_logging(level="INFO", log_file=self.temp_file)
        
        # Create tracker
        tracker = LLMUsageTracker()
        
        # Track failed usage - should log error
        tracker.track_usage(
            model="gpt-3.5-turbo",
            provider="openai",
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            cost=0.0,
            success=False,
            error_message="Rate limit exceeded"
        )
        
        # Check that error log was written
        with open(self.temp_file, 'r') as f:
            log_content = f.read()
        self.assertIn("Tracked failed usage", log_content)
        self.assertIn("Rate limit exceeded", log_content)
    
    def test_client_logging(self):
        """Test logging in LLM client."""
        # Set up logging
        setup_logging(level="INFO", log_file=self.temp_file)
        
        # Create client
        client = LLMClient(default_model="gpt-3.5-turbo")
        
        # Mock the completion call
        with patch('llm_utils.llm_util.completion') as mock_completion, \
             patch('llm_utils.llm_util.litellm_get_total_cost') as mock_completion_cost, \
             patch('llm_utils.llm_util.token_counter') as mock_token_counter, \
             patch('llm_utils.llm_util.get_llm_provider') as mock_get_provider:
            
            mock_get_provider.return_value = "openai"
            mock_completion.return_value = {
                'choices': [{'message': {'content': 'Hello!'}}],
                'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}
            }
            mock_completion_cost.return_value = 0.001  # Now returns float
            mock_token_counter.side_effect = [10, 5]  # prompt_tokens, completion_tokens
            
            # Make request
            response = client.chat_completion([
                {"role": "user", "content": "Hello"}
            ])
            
            # Check that logs were written
            with open(self.temp_file, 'r') as f:
                log_content = f.read()
            
            self.assertIn("Making LLM chat completion to gpt-3.5-turbo", log_content)
            self.assertIn("LLM chat completion successful", log_content)
            self.assertIn("openai", log_content)
    
    def test_client_debug_logging(self):
        """Test debug logging in LLM client."""
        # Set up debug logging
        setup_logging(level="DEBUG", log_file=self.temp_file)
        
        # Create client
        client = LLMClient(default_model="gpt-3.5-turbo")
        
        # Mock the completion call
        with patch('llm_utils.llm_util.completion') as mock_completion, \
             patch('llm_utils.llm_util.litellm_get_total_cost') as mock_completion_cost, \
             patch('llm_utils.llm_util.token_counter') as mock_token_counter, \
             patch('llm_utils.llm_util.get_llm_provider') as mock_get_provider:
            
            mock_get_provider.return_value = "openai"
            mock_completion.return_value = {
                'choices': [{'message': {'content': 'Hello world!'}}],
                'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}
            }
            mock_completion_cost.return_value = 0.001  # Now returns float
            mock_token_counter.side_effect = [10, 5]  # prompt_tokens, completion_tokens
            
            # Make request
            response = client.chat_completion([
                {"role": "user", "content": "Hello"}
            ])
            
            # Check that debug logs were written
            with open(self.temp_file, 'r') as f:
                log_content = f.read()
            
            self.assertIn("Request messages:", log_content)
            self.assertIn("Request parameters:", log_content)
            self.assertIn("Response content:", log_content)
    
    def test_client_error_logging(self):
        """Test error logging in LLM client."""
        # Set up logging
        setup_logging(level="INFO", log_file=self.temp_file)
        
        # Create client
        client = LLMClient(default_model="gpt-3.5-turbo")
        
        # Mock the completion call to raise an error
        with patch('llm_utils.llm_util.completion') as mock_completion, \
             patch('llm_utils.llm_util.get_llm_provider') as mock_get_provider:
            
            mock_get_provider.return_value = "openai"
            mock_completion.side_effect = Exception("API Error")
            
            # Make request - should log error
            with self.assertRaises(Exception):
                client.chat_completion([
                    {"role": "user", "content": "Hello"}
                ])
            
            # Check that error log was written
            with open(self.temp_file, 'r') as f:
                log_content = f.read()
            
            self.assertIn("Unexpected error in LLM chat completion", log_content)
            self.assertIn("API Error", log_content)
    
    def test_client_embedding_logging(self):
        """Test logging in LLM client for embeddings."""
        # Set up logging
        setup_logging(level="INFO", log_file=self.temp_file)
        
        # Create client
        client = LLMClient(default_model="text-embedding-ada-002")
        
        # Mock the embedding call
        with patch('llm_utils.llm_util.embedding') as mock_embedding, \
             patch('llm_utils.llm_util.litellm_get_total_cost') as mock_embedding_cost, \
             patch('llm_utils.llm_util.token_counter') as mock_token_counter, \
             patch('llm_utils.llm_util.get_llm_provider') as mock_get_provider:
            
            mock_get_provider.return_value = "openai"
            mock_embedding.return_value = {
                'data': [{'embedding': [0.1, 0.2, 0.3]}],
                'usage': {'total_tokens': 5, 'prompt_tokens': 5}
            }
            mock_embedding_cost.return_value = 0.0001  # Now returns float
            mock_token_counter.side_effect = [5]  # prompt_tokens
            
            # Make request
            response = client.embedding(input="Hello world")
            
            # Check that logs were written
            with open(self.temp_file, 'r') as f:
                log_content = f.read()
            
            self.assertIn("Making LLM embedding to text-embedding-ada-002", log_content)
            self.assertIn("LLM embedding successful", log_content)
            self.assertIn("openai", log_content)
    
    def test_client_embedding_debug_logging(self):
        """Test debug logging in LLM client for embeddings."""
        # Set up debug logging
        setup_logging(level="DEBUG", log_file=self.temp_file)
        
        # Create client
        client = LLMClient(default_model="text-embedding-ada-002")
        
        # Mock the embedding call
        with patch('llm_utils.llm_util.embedding') as mock_embedding, \
             patch('llm_utils.llm_util.litellm_get_total_cost') as mock_embedding_cost, \
             patch('llm_utils.llm_util.token_counter') as mock_token_counter, \
             patch('llm_utils.llm_util.get_llm_provider') as mock_get_provider:
            
            mock_get_provider.return_value = "openai"
            mock_embedding.return_value = {
                'data': [{'embedding': [0.1, 0.2, 0.3]}],
                'usage': {'total_tokens': 5, 'prompt_tokens': 5}
            }
            mock_embedding_cost.return_value = 0.0001  # Now returns float
            mock_token_counter.side_effect = [5]  # prompt_tokens
            
            # Make request
            response = client.embedding(input=["Hello", "World"])
            
            # Check that debug logs were written
            with open(self.temp_file, 'r') as f:
                log_content = f.read()
            
            self.assertIn("Request embeddings for 2 input(s)", log_content)
            self.assertIn("Response: 1 embedding(s) of dimension 3", log_content)
    
    def test_retry_logging(self):
        """Test retry logging in LLM client."""
        # Set up logging
        setup_logging(level="INFO", log_file=self.temp_file)
        
        # Create client with short retry delay for testing
        client = LLMClient(default_model="gpt-3.5-turbo", default_retry_delay=0.1)
        
        # Mock the completion call to fail first, then succeed
        with patch('llm_utils.llm_util.completion') as mock_completion, \
             patch('llm_utils.llm_util.litellm_get_total_cost') as mock_completion_cost, \
             patch('llm_utils.llm_util.token_counter') as mock_token_counter, \
             patch('llm_utils.llm_util.get_llm_provider') as mock_get_provider:
            
            mock_get_provider.return_value = "openai"
            
            # Import the exception class
            try:
                from litellm.exceptions import RateLimitError
            except ImportError:
                # Create a mock exception if litellm is not available
                class RateLimitError(Exception):
                    pass
            
            # First call fails with rate limit, second succeeds
            mock_completion.side_effect = [
                RateLimitError(500, "Rate limit exceeded", "openai", "gpt-3.5-turbo"),
                {'choices': [{'message': {'content': 'Success!'}}], 'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}}
            ]
            
            mock_completion_cost.return_value = 0.001  # Now returns float
            mock_token_counter.side_effect = [10, 5]  # prompt_tokens, completion_tokens
            
            # Make request
            response = client.chat_completion([
                {"role": "user", "content": "Hello"}
            ])
            
            # Check that retry logs were written
            with open(self.temp_file, 'r') as f:
                log_content = f.read()
            
            self.assertIn("Retry attempt", log_content)
            self.assertIn("Retrying in", log_content)
    
    def test_custom_logger(self):
        """Test using custom logger with client."""
        # Create custom logger
        custom_logger = logging.getLogger("custom_test_logger")
        custom_logger.setLevel(logging.INFO)
        
        # Add handler to custom logger
        handler = logging.FileHandler(self.temp_file)
        formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        custom_logger.addHandler(handler)
        
        # Create client with custom logger
        client = LLMClient(default_model="gpt-3.5-turbo", logger=custom_logger)
        
        # Mock the completion call
        with patch('llm_utils.llm_util.completion') as mock_completion, \
             patch('llm_utils.llm_util.litellm_get_total_cost') as mock_completion_cost, \
             patch('llm_utils.llm_util.token_counter') as mock_token_counter, \
             patch('llm_utils.llm_util.get_llm_provider') as mock_get_provider:
            
            mock_get_provider.return_value = "openai"
            mock_completion.return_value = {
                'choices': [{'message': {'content': 'Hello!'}}],
                'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}
            }
            mock_completion_cost.return_value = 0.001  # Now returns float
            mock_token_counter.side_effect = [10, 5]  # prompt_tokens, completion_tokens
            
            # Make request
            response = client.chat_completion([
                {"role": "user", "content": "Hello"}
            ])
            
            # Check that custom logger was used
            with open(self.temp_file, 'r') as f:
                log_content = f.read()
            
            self.assertIn("custom_test_logger", log_content)
            self.assertIn("Making LLM chat completion", log_content)


class TestUsageCheckpointsTracker(unittest.TestCase):
    """Tests for checkpoint functionality on LLMUsageTracker."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_file = os.path.join(self.temp_dir, "checkpoint_usage.json")
        self.tracker = LLMUsageTracker(self.temp_file)
    
    def tearDown(self):
        if os.path.exists(self.temp_file):
            os.remove(self.temp_file)
        os.rmdir(self.temp_dir)
    
    def test_basic_checkpoint(self):
        """Start/end a single checkpoint and verify aggregated stats."""
        self.tracker.start_usage_checkpoint("A")
        # Two successful events inside A
        self.tracker.track_usage("gpt-4", "openai", 10, 5, 15, 0.001, response_time=1.0)
        self.tracker.track_usage("gpt-3.5-turbo", "openai", 8, 4, 12, 0.0008, response_time=0.8)
        # One event outside A
        self.tracker.end_usage_checkpoint("A")
        self.tracker.track_usage("claude-3", "anthropic", 9, 3, 12, 0.0009, response_time=1.2)
        
        agg = self.tracker.get_checkpoint_usage("A")
        self.assertEqual(agg.total_requests, 2)
        self.assertEqual(agg.total_tokens, 27)
        self.assertAlmostEqual(agg.total_cost, 0.0018, places=6)
        self.assertIn("gpt-4", agg.model_breakdown)
        self.assertIn("gpt-3.5-turbo", agg.model_breakdown)
        self.assertNotIn("claude-3", agg.model_breakdown)
    
    def test_nested_and_repeated_checkpoints(self):
        """Support nested checkpoints and multiple start/end cycles for the same name."""
        # Outer begins
        self.tracker.start_usage_checkpoint("outer")
        self.tracker.track_usage("gpt-4", "openai", 10, 0, 10, 0.0005, response_time=0.5)
        
        # Inner nested
        self.tracker.start_usage_checkpoint("inner")
        self.tracker.track_usage("gpt-4", "openai", 5, 5, 10, 0.0007, response_time=0.6)
        self.tracker.track_usage("claude-3", "anthropic", 6, 4, 10, 0.0006, response_time=0.7)
        self.tracker.end_usage_checkpoint("inner")
        
        # Back to outer only
        self.tracker.track_usage("gpt-3.5-turbo", "openai", 4, 3, 7, 0.0003, response_time=0.4)
        self.tracker.end_usage_checkpoint("outer")
        
        # Re-open outer (repeated name) for additional interval
        self.tracker.start_usage_checkpoint("outer")
        self.tracker.track_usage("gpt-4", "openai", 3, 2, 5, 0.0002, response_time=0.3)
        self.tracker.end_usage_checkpoint("outer")
        
        inner_agg = self.tracker.get_checkpoint_usage("inner")
        self.assertEqual(inner_agg.total_requests, 2)
        self.assertEqual(inner_agg.total_tokens, 20)
        
        outer_agg = self.tracker.get_checkpoint_usage("outer")
        # First outer interval has 1 + 2 + 1 = 4 events, second interval has 1 event => 5 total
        self.assertEqual(outer_agg.total_requests, 5)
        self.assertEqual(outer_agg.total_tokens, 10 + 10 + 10 + 7 + 5)


class TestUsageCheckpointsClient(unittest.TestCase):
    """Tests for checkpoint functionality via LLMClient convenience methods."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_file = os.path.join(self.temp_dir, "checkpoint_client_usage.json")
        self.tracker = LLMUsageTracker(self.temp_file)
        self.client = LLMClient(self.tracker)
    
    def tearDown(self):
        if os.path.exists(self.temp_file):
            os.remove(self.temp_file)
        os.rmdir(self.temp_dir)
    
    def test_client_checkpoint_summary(self):
        """Use client API to start/end and get formatted summary dict."""
        self.client.start_usage_checkpoint("part1")
        # Add usage directly to underlying tracker; checkpointing is index-based
        self.client.usage_tracker.track_usage("gpt-4", "openai", 12, 3, 15, 0.0009, response_time=0.9)
        self.client.usage_tracker.track_usage("gpt-3.5-turbo", "openai", 7, 2, 9, 0.0004, response_time=0.7)
        self.client.end_usage_checkpoint("part1")
        # Outside checkpoint
        self.client.usage_tracker.track_usage("claude-3", "anthropic", 5, 5, 10, 0.0008, response_time=0.8)
        
        summary = self.client.get_checkpoint_usage("part1")
        self.assertEqual(summary["total_requests"], 2)
        self.assertEqual(summary["total_tokens"], 24)
        self.assertTrue(summary["total_cost"].startswith("$"))
        self.assertIn("top_models", summary)
        self.assertIn("top_providers", summary)
        
    def test_open_checkpoint_included_until_now(self):
        """An active checkpoint should include events up to the current moment."""
        self.client.start_usage_checkpoint("open")
        self.client.usage_tracker.track_usage("gpt-4", "openai", 2, 2, 4, 0.0001, response_time=0.2)
        # Do not end; query while open
        summary = self.client.get_checkpoint_usage("open")
        self.assertEqual(summary["total_requests"], 1)
        self.assertEqual(summary["total_tokens"], 4)
        # End and confirm still at least those events
        self.client.end_usage_checkpoint("open")
        summary2 = self.client.get_checkpoint_usage("open")
        self.assertGreaterEqual(summary2["total_requests"], 1)
        self.assertGreaterEqual(summary2["total_tokens"], 4)


class TestCheckpointSaveLoadExport(unittest.TestCase):
    """Tests for checkpoint save/load/export functionality."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_file = os.path.join(self.temp_dir, "checkpoint_save_test.json")
        self.tracker = LLMUsageTracker(self.temp_file)
    
    def tearDown(self):
        if os.path.exists(self.temp_file):
            os.remove(self.temp_file)
        os.rmdir(self.temp_dir)
    
    def test_save_checkpoint_data(self):
        """Test saving usage data with checkpoint information."""
        # Create some usage data with checkpoints
        self.tracker.track_usage("gpt-4", "openai", 10, 5, 15, 0.001, response_time=1.0)
        
        self.tracker.start_usage_checkpoint("test_checkpoint")
        self.tracker.track_usage("gpt-3.5-turbo", "openai", 8, 4, 12, 0.0008, response_time=0.8)
        self.tracker.track_usage("claude-3", "anthropic", 6, 3, 9, 0.0006, response_time=0.9)
        self.tracker.end_usage_checkpoint("test_checkpoint")
        
        self.tracker.track_usage("gpt-4", "openai", 5, 2, 7, 0.0005, response_time=0.7)
        
        # Save data
        self.tracker.save_usage_data()
        
        # Verify file was created and contains checkpoint info
        self.assertTrue(os.path.exists(self.temp_file))
        
        with open(self.temp_file, 'r') as f:
            data = json.load(f)
        
        # Check structure
        self.assertIn("usage_data", data)
        self.assertIn("checkpoint_ranges", data)
        self.assertIn("checkpoint_stacks", data)
        self.assertIn("metadata", data)
        
        # Check usage data
        self.assertEqual(len(data["usage_data"]), 4)
        
        # Check checkpoint ranges
        self.assertIn("test_checkpoint", data["checkpoint_ranges"])
        self.assertEqual(len(data["checkpoint_ranges"]["test_checkpoint"]), 1)
        self.assertEqual(data["checkpoint_ranges"]["test_checkpoint"][0], [1, 3])  # indices 1-2
        
        # Check metadata
        self.assertIn("saved_at", data["metadata"])
        self.assertEqual(data["metadata"]["total_usage_records"], 4)
        self.assertEqual(data["metadata"]["total_checkpoints"], 1)
    
    def test_load_checkpoint_data(self):
        """Test loading usage data with checkpoint information."""
        # Create and save data with checkpoints
        self.tracker.track_usage("gpt-4", "openai", 10, 5, 15, 0.001, response_time=1.0)
        
        self.tracker.start_usage_checkpoint("test_checkpoint")
        self.tracker.track_usage("gpt-3.5-turbo", "openai", 8, 4, 12, 0.0008, response_time=0.8)
        self.tracker.end_usage_checkpoint("test_checkpoint")
        
        self.tracker.save_usage_data()
        
        # Create new tracker and load data
        new_tracker = LLMUsageTracker(os.path.join(self.temp_dir, "new_checkpoint_test.json"))
        new_tracker.load_usage_data(self.temp_file)
        
        # Verify data was loaded
        self.assertEqual(len(new_tracker.usage_data), 2)
        self.assertEqual(new_tracker.usage_data[0].model, "gpt-4")
        self.assertEqual(new_tracker.usage_data[1].model, "gpt-3.5-turbo")
        
        # Verify checkpoint information was restored
        self.assertIn("test_checkpoint", new_tracker._checkpoint_ranges)
        self.assertEqual(len(new_tracker._checkpoint_ranges["test_checkpoint"]), 1)
        self.assertEqual(new_tracker._checkpoint_ranges["test_checkpoint"][0], (1, 2))
        
        # Verify checkpoint usage works
        checkpoint_usage = new_tracker.get_checkpoint_usage("test_checkpoint")
        self.assertEqual(checkpoint_usage.total_requests, 1)
        self.assertEqual(checkpoint_usage.total_tokens, 12)
    
    def test_export_checkpoint_data(self):
        """Test exporting data for a specific checkpoint."""
        # Create usage data with checkpoints
        self.tracker.track_usage("gpt-4", "openai", 10, 5, 15, 0.001, response_time=1.0)
        
        self.tracker.start_usage_checkpoint("test_checkpoint")
        self.tracker.track_usage("gpt-3.5-turbo", "openai", 8, 4, 12, 0.0008, response_time=0.8)
        self.tracker.track_usage("claude-3", "anthropic", 6, 3, 9, 0.0006, response_time=0.9)
        self.tracker.end_usage_checkpoint("test_checkpoint")
        
        self.tracker.track_usage("gpt-4", "openai", 5, 2, 7, 0.0005, response_time=0.7)
        
        # Export all data
        all_export_file = os.path.join(self.temp_dir, "all_data.json")
        self.tracker.export_usage_data(all_export_file, format='json')
        
        with open(all_export_file, 'r') as f:
            all_data = json.load(f)
        self.assertEqual(len(all_data), 4)
        
        # Export checkpoint data only
        checkpoint_export_file = os.path.join(self.temp_dir, "checkpoint_data.json")
        self.tracker.export_usage_data(checkpoint_export_file, format='json', checkpoint_name="test_checkpoint")
        
        with open(checkpoint_export_file, 'r') as f:
            checkpoint_data = json.load(f)
        self.assertEqual(len(checkpoint_data), 2)
        self.assertEqual(checkpoint_data[0]["model"], "gpt-3.5-turbo")
        self.assertEqual(checkpoint_data[1]["model"], "claude-3")
        
        # Clean up
        os.remove(all_export_file)
        os.remove(checkpoint_export_file)
    
    def test_export_checkpoint_csv(self):
        """Test exporting checkpoint data as CSV."""
        # Create usage data with checkpoints
        self.tracker.start_usage_checkpoint("test_checkpoint")
        self.tracker.track_usage("gpt-3.5-turbo", "openai", 8, 4, 12, 0.0008, response_time=0.8)
        self.tracker.track_usage("claude-3", "anthropic", 6, 3, 9, 0.0006, response_time=0.9)
        self.tracker.end_usage_checkpoint("test_checkpoint")
        
        # Export as CSV
        csv_file = os.path.join(self.temp_dir, "checkpoint_data.csv")
        self.tracker.export_usage_data(csv_file, format='csv', checkpoint_name="test_checkpoint")
        
        # Verify CSV file
        self.assertTrue(os.path.exists(csv_file))
        
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["model"], "gpt-3.5-turbo")
        self.assertEqual(rows[1]["model"], "claude-3")
        
        # Clean up
        os.remove(csv_file)
    
    def test_export_nonexistent_checkpoint(self):
        """Test exporting data for a nonexistent checkpoint."""
        # Create some usage data
        self.tracker.track_usage("gpt-4", "openai", 10, 5, 15, 0.001, response_time=1.0)
        
        # Export nonexistent checkpoint
        export_file = os.path.join(self.temp_dir, "nonexistent_checkpoint.json")
        self.tracker.export_usage_data(export_file, format='json', checkpoint_name="nonexistent")
        
        # Should create empty file
        with open(export_file, 'r') as f:
            data = json.load(f)
        self.assertEqual(len(data), 0)
        
        # Clean up
        os.remove(export_file)
    
    def test_client_export_checkpoint(self):
        """Test exporting checkpoint data through client."""
        client = LLMClient(self.tracker)
        
        # Create usage data with checkpoints
        client.start_usage_checkpoint("test_checkpoint")
        client.usage_tracker.track_usage("gpt-3.5-turbo", "openai", 8, 4, 12, 0.0008, response_time=0.8)
        client.end_usage_checkpoint("test_checkpoint")
        
        # Export through client
        export_file = os.path.join(self.temp_dir, "client_checkpoint_export.json")
        client.export_usage_data(export_file, format='json', checkpoint_name="test_checkpoint")
        
        # Verify export
        with open(export_file, 'r') as f:
            data = json.load(f)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["model"], "gpt-3.5-turbo")
        
        # Clean up
        os.remove(export_file)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)
