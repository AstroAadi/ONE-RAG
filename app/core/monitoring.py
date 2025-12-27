"""
Monitoring and Metrics Collection
Tracks performance, errors, and system health
"""

import time
import logging
import sys
from typing import Optional, Dict, Any, Callable
from functools import wraps
from datetime import datetime, timedelta
from collections import defaultdict, deque
from pathlib import Path
import threading

# ==================== Logging Setup ====================

LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    console: bool = True
) -> None:
    """
    Setup logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for logging
        console: Whether to log to console
    """
    handlers = []
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
        handlers.append(console_handler)
    
    # File handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,
        handlers=handlers,
        force=True
    )


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance
    
    Args:
        name: Logger name (usually __name__)
    
    Returns:
        Configured logger instance
    
    Usage:
        logger = get_logger(__name__)
        logger.info("Message")
    """
    return logging.getLogger(name or __name__)


# ==================== Metrics Collection ====================

class MetricsCollector:
    """
    Thread-safe metrics collector for monitoring system performance
    
    Tracks:
    - Request counts
    - Processing times
    - Error rates
    - Resource usage
    """
    
    def __init__(self, window_minutes: int = 60):
        self.window_minutes = window_minutes
        self._lock = threading.Lock()
        
        # Metrics storage
        self._request_counts: Dict[str, int] = defaultdict(int)
        self._error_counts: Dict[str, int] = defaultdict(int)
        self._processing_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._timestamps: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # System metrics
        self.total_requests = 0
        self.total_errors = 0
        self.start_time = datetime.utcnow()
    
    def record_request(self, endpoint: str) -> None:
        """Record a request to an endpoint"""
        with self._lock:
            self._request_counts[endpoint] += 1
            self.total_requests += 1
            self._timestamps[endpoint].append(datetime.utcnow())
    
    def record_error(self, endpoint: str, error_type: str) -> None:
        """Record an error"""
        with self._lock:
            error_key = f"{endpoint}:{error_type}"
            self._error_counts[error_key] += 1
            self.total_errors += 1
    
    def record_processing_time(self, endpoint: str, duration_seconds: float) -> None:
        """Record processing time for an endpoint"""
        with self._lock:
            self._processing_times[endpoint].append(duration_seconds)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics summary"""
        with self._lock:
            uptime = datetime.utcnow() - self.start_time
            
            metrics = {
                "uptime_seconds": uptime.total_seconds(),
                "total_requests": self.total_requests,
                "total_errors": self.total_errors,
                "error_rate": (
                    self.total_errors / self.total_requests 
                    if self.total_requests > 0 else 0
                ),
                "endpoints": {}
            }
            
            # Per-endpoint metrics
            for endpoint, times in self._processing_times.items():
                if times:
                    metrics["endpoints"][endpoint] = {
                        "request_count": self._request_counts[endpoint],
                        "avg_time_seconds": sum(times) / len(times),
                        "min_time_seconds": min(times),
                        "max_time_seconds": max(times),
                        "p50_time_seconds": self._percentile(times, 50),
                        "p95_time_seconds": self._percentile(times, 95),
                        "p99_time_seconds": self._percentile(times, 99)
                    }
            
            # Error breakdown
            if self._error_counts:
                metrics["errors"] = dict(self._error_counts)
            
            return metrics
    
    def get_recent_requests_per_minute(self, endpoint: Optional[str] = None) -> float:
        """Get recent requests per minute"""
        with self._lock:
            cutoff = datetime.utcnow() - timedelta(minutes=self.window_minutes)
            
            if endpoint:
                timestamps = self._timestamps.get(endpoint, [])
            else:
                # Aggregate all endpoints
                timestamps = []
                for ts_list in self._timestamps.values():
                    timestamps.extend(ts_list)
            
            recent = [ts for ts in timestamps if ts > cutoff]
            
            if not recent:
                return 0.0
            
            minutes_elapsed = max(1, self.window_minutes)
            return len(recent) / minutes_elapsed
    
    @staticmethod
    def _percentile(data: deque, percentile: int) -> float:
        """Calculate percentile of data"""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def reset_metrics(self) -> None:
        """Reset all metrics"""
        with self._lock:
            self._request_counts.clear()
            self._error_counts.clear()
            self._processing_times.clear()
            self._timestamps.clear()
            self.total_requests = 0
            self.total_errors = 0
            self.start_time = datetime.utcnow()


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None
_logger = get_logger(__name__)


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def track_performance(endpoint_name: str):
    """
    Decorator to track function performance
    
    Usage:
        @track_performance("upload_file")
        def upload_file(...):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            metrics = get_metrics_collector()
            metrics.record_request(endpoint_name)
            
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                metrics.record_processing_time(endpoint_name, duration)
                
                _logger.debug(f"{endpoint_name} completed in {duration:.3f}s")
                return result
            except Exception as e:
                duration = time.time() - start_time
                metrics.record_processing_time(endpoint_name, duration)
                metrics.record_error(endpoint_name, type(e).__name__)
                
                _logger.error(
                    f"{endpoint_name} failed after {duration:.3f}s: {type(e).__name__}"
                )
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            metrics = get_metrics_collector()
            metrics.record_request(endpoint_name)
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                metrics.record_processing_time(endpoint_name, duration)
                
                _logger.debug(f"{endpoint_name} completed in {duration:.3f}s")
                return result
            except Exception as e:
                duration = time.time() - start_time
                metrics.record_processing_time(endpoint_name, duration)
                metrics.record_error(endpoint_name, type(e).__name__)
                
                _logger.error(
                    f"{endpoint_name} failed after {duration:.3f}s: {type(e).__name__}"
                )
                raise
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


class PerformanceTimer:
    """
    Context manager for timing code blocks
    
    Usage:
        with PerformanceTimer("embedding_generation") as timer:
            embeddings = generate_embeddings(texts)
        
        logger.info(f"Embeddings took {timer.duration:.3f}s")
    """
    
    def __init__(self, operation_name: str, log_result: bool = True):
        self.operation_name = operation_name
        self.log_result = log_result
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.duration: Optional[float] = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        
        if self.log_result:
            if exc_type is None:
                _logger.debug(
                    f"⏱️  {self.operation_name} completed in {self.duration:.3f}s"
                )
            else:
                _logger.error(
                    f"❌ {self.operation_name} failed after {self.duration:.3f}s: "
                    f"{exc_type.__name__}"
                )


def log_function_call(func: Callable) -> Callable:
    """
    Decorator to log function calls with arguments
    
    Usage:
        @log_function_call
        def process_document(file_id: str, user_id: str):
            ...
    """
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        func_name = func.__name__
        _logger.debug(f"→ Calling {func_name} with args={args}, kwargs={kwargs}")
        
        try:
            result = await func(*args, **kwargs)
            _logger.debug(f"✓ {func_name} completed successfully")
            return result
        except Exception as e:
            _logger.error(f"✗ {func_name} failed: {type(e).__name__}: {e}")
            raise
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        func_name = func.__name__
        _logger.debug(f"→ Calling {func_name} with args={args}, kwargs={kwargs}")
        
        try:
            result = func(*args, **kwargs)
            _logger.debug(f"✓ {func_name} completed successfully")
            return result
        except Exception as e:
            _logger.error(f"✗ {func_name} failed: {type(e).__name__}: {e}")
            raise
    
    import asyncio
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper


def print_metrics_summary():
    """Print metrics summary to console"""
    metrics = get_metrics_collector().get_metrics()
    
    print("\n" + "="*60)
    print("SYSTEM METRICS")
    print("="*60)
    
    print(f"\n📊 Overall:")
    print(f"  - Uptime: {metrics['uptime_seconds']:.0f} seconds")
    print(f"  - Total Requests: {metrics['total_requests']}")
    print(f"  - Total Errors: {metrics['total_errors']}")
    print(f"  - Error Rate: {metrics['error_rate']:.2%}")
    
    if metrics.get('endpoints'):
        print(f"\n🎯 Endpoints:")
        for endpoint, stats in metrics['endpoints'].items():
            print(f"\n  {endpoint}:")
            print(f"    - Requests: {stats['request_count']}")
            print(f"    - Avg Time: {stats['avg_time_seconds']:.3f}s")
            print(f"    - P95 Time: {stats['p95_time_seconds']:.3f}s")
            print(f"    - P99 Time: {stats['p99_time_seconds']:.3f}s")
    
    if metrics.get('errors'):
        print(f"\n❌ Errors:")
        for error_key, count in metrics['errors'].items():
            print(f"  - {error_key}: {count}")
    
    print("="*60 + "\n")