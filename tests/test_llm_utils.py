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
    create_llm_client, get_usage_summary
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
            error_message=None
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
    
    def test_client_initialization_defaults(self):
        """Test client initialization with default tracker."""
        client = LLMClient()
        self.assertIsInstance(client.usage_tracker, LLMUsageTracker)
    
    @patch('llm_utils.llm_util.completion')
    @patch('llm_utils.llm_util.completion_cost')
    @patch('llm_utils.llm_util.get_llm_provider')
    def test_chat_completion_success(self, mock_get_provider, mock_completion_cost, mock_completion):
        """Test successful chat completion."""
        # Mock responses
        mock_get_provider.return_value = "openai"
        mock_completion.return_value = {
            'choices': [{'message': {'content': 'Hello!'}}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}
        }
        mock_completion_cost.return_value = {
            'prompt_tokens': 10,
            'completion_tokens': 5,
            'total_tokens': 15,
            'cost': 0.001
        }
        
        messages = [{"role": "user", "content": "Hello"}]
        response = self.client.chat_completion("gpt-3.5-turbo", messages)
        
        self.assertEqual(response['choices'][0]['message']['content'], 'Hello!')
        mock_completion.assert_called_once()
        
        # Check that usage was tracked
        self.assertEqual(len(self.tracker.usage_data), 1)
        usage = self.tracker.usage_data[0]
        self.assertEqual(usage.model, "gpt-3.5-turbo")
        self.assertEqual(usage.provider, "openai")
        self.assertTrue(usage.success)
    
    @patch('llm_utils.llm_util.completion')
    @patch('llm_utils.llm_util.completion_cost')
    @patch('llm_utils.llm_util.get_llm_provider')
    def test_text_completion_success(self, mock_get_provider, mock_completion_cost, mock_completion):
        """Test successful text completion."""
        # Mock responses
        mock_get_provider.return_value = "openai"
        mock_completion.return_value = {
            'choices': [{'text': 'Hello world!'}],
            'usage': {'prompt_tokens': 5, 'completion_tokens': 3, 'total_tokens': 8}
        }
        mock_completion_cost.return_value = {
            'prompt_tokens': 5,
            'completion_tokens': 3,
            'total_tokens': 8,
            'cost': 0.0005
        }
        
        response = self.client.text_completion("gpt-3.5-turbo", "Hello")
        
        self.assertEqual(response['choices'][0]['text'], 'Hello world!')
        mock_completion.assert_called_once()
        
        # Check that usage was tracked
        self.assertEqual(len(self.tracker.usage_data), 1)
        usage = self.tracker.usage_data[0]
        self.assertEqual(usage.model, "gpt-3.5-turbo")
        self.assertTrue(usage.success)
    
    @patch('llm_utils.llm_util.completion')
    @patch('llm_utils.llm_util.completion_cost')
    @patch('llm_utils.llm_util.get_llm_provider')
    def test_retry_logic_on_rate_limit(self, mock_get_provider, mock_completion_cost, mock_completion):
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
        
        with patch('llm_utils.llm_util.completion_cost') as mock_completion_cost:
            mock_completion_cost.return_value = {
                'prompt_tokens': 10,
                'completion_tokens': 5,
                'total_tokens': 15,
                'cost': 0.001
            }
            
            messages = [{"role": "user", "content": "Hello"}]
            response = self.client.chat_completion("gpt-3.5-turbo", messages)

            self.assertEqual(response['choices'][0]['message']['content'], 'Success!')
            self.assertEqual(mock_completion.call_count, 2)
    
    @patch('llm_utils.llm_util.completion')
    @patch('llm_utils.llm_util.completion_cost')
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
            self.client.chat_completion("gpt-3.5-turbo", messages)
        
        # Check that failed usage was tracked
        self.assertEqual(len(self.tracker.usage_data), 1)
        usage = self.tracker.usage_data[0]
        self.assertFalse(usage.success)
        self.assertIn("API error", usage.error_message)
    
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
    
    def test_create_llm_client_default(self):
        """Test create_llm_client function with default file."""
        client = create_llm_client()
        
        self.assertIsInstance(client, LLMClient)
        self.assertEqual(client.usage_tracker.data_file, "llm_usage_data.json")
    
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


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)
