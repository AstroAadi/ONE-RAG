"""
Enhanced Task Manager with Progress Tracking and Better Concurrency
Fixes blocking issues and adds real-time progress updates
"""

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future
from typing import Dict, Optional, Callable, Any
import logging
from enum import Enum
from datetime import datetime
import uuid
import threading

logger = logging.getLogger(__name__)


class TaskType(str, Enum):
    """Task type classification"""
    CPU_BOUND = "cpu_bound"  # PDF extraction, image processing
    IO_BOUND = "io_bound"    # Embeddings, vector store, database locust ade 


class TaskStatus(str, Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ProcessingStage(str, Enum):
    """Document processing stages for progress tracking"""
    QUEUED = "queued"
    EXTRACTING = "extracting"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    FINALIZING = "finalizing"
    COMPLETE = "complete"
    FAILED = "failed"


class TaskInfo:
    """Container for task information with progress tracking"""
    
    def __init__(
        self,
        task_id: str,
        task_type: TaskType,
        description: str,
        user_id: str,
        file_id: Optional[str] = None
    ):
        self.task_id = task_id
        self.task_type = task_type
        self.description = description
        self.user_id = user_id
        self.file_id = file_id
        self.status = TaskStatus.PENDING
        self.progress = 0.0
        self.result = None
        self.error = None
        self.created_at = datetime.utcnow()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.future: Optional[Future] = None
        
        # NEW: Progress tracking
        self.current_stage = ProcessingStage.QUEUED
        self.stage_details = ""
        self.chunks_created = 0
        self.total_chunks = 0
        self.pages_processed = 0
        self.total_pages = 0
        
        # Thread-safe lock for updates
        self._lock = threading.Lock()
    
    def update_progress(
        self,
        progress: float,
        stage: Optional[ProcessingStage] = None,
        details: Optional[str] = None,
        chunks_created: Optional[int] = None,
        total_chunks: Optional[int] = None,
        pages_processed: Optional[int] = None,
        total_pages: Optional[int] = None
    ):
        """Thread-safe progress update"""
        with self._lock:
            self.progress = max(0.0, min(100.0, progress))
            if stage:
                self.current_stage = stage
            if details:
                self.stage_details = details
            if chunks_created is not None:
                self.chunks_created = chunks_created
            if total_chunks is not None:
                self.total_chunks = total_chunks
            if pages_processed is not None:
                self.pages_processed = pages_processed
            if total_pages is not None:
                self.total_pages = total_pages
            
            logger.info(
                f"Task {self.task_id}: {self.progress:.1f}% - "
                f"{stage.value if stage else self.current_stage.value} - {details or self.stage_details}"
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        with self._lock:
            return {
                "task_id": self.task_id,
                "task_type": self.task_type.value,
                "description": self.description,
                "user_id": self.user_id,
                "file_id": self.file_id,
                "status": self.status.value,
                "progress": round(self.progress, 1),
                "current_stage": self.current_stage.value,
                "stage_details": self.stage_details,
                "chunks_created": self.chunks_created,
                "total_chunks": self.total_chunks,
                "pages_processed": self.pages_processed,
                "total_pages": self.total_pages,
                "error": self.error,
                "created_at": self.created_at.isoformat(),
                "started_at": self.started_at.isoformat() if self.started_at else None,
                "completed_at": self.completed_at.isoformat() if self.completed_at else None,
                "processing_time": self._get_processing_time()
            }
    
    def _get_processing_time(self) -> Optional[float]:
        """Calculate processing time in seconds"""
        if self.completed_at and self.started_at:
            return (self.completed_at - self.started_at).total_seconds()
        elif self.started_at:
            return (datetime.utcnow() - self.started_at).total_seconds()
        return None


class TaskManager:
    """
    Enhanced task manager with progress tracking
    
    Key improvements:
    - Progress callbacks for real-time updates
    - Better separation of CPU vs I/O tasks
    - Thread-safe progress updates
    - Detailed stage tracking
    """
    
    def __init__(
        self,
        max_io_workers: int = 20,
        max_cpu_workers: int = 4
    ):
        """Initialize task manager with worker pools"""
        self.io_executor = ThreadPoolExecutor(
            max_workers=max_io_workers,
            thread_name_prefix="io_worker"
        )
        
        self.cpu_executor = ProcessPoolExecutor(
            max_workers=max_cpu_workers
        )
        
        # Task tracking (thread-safe dict)
        self.tasks: Dict[str, TaskInfo] = {}
        self._tasks_lock = threading.Lock()
        
        logger.info(
            f"✅ Task Manager initialized: "
            f"I/O workers={max_io_workers}, CPU workers={max_cpu_workers}"
        )
    
    def submit_task(
        self,
        func: Callable,
        task_type: TaskType,
        description: str,
        user_id: str,
        file_id: Optional[str] = None,
        *args,
        **kwargs
    ) -> str:
        """
        Submit a task for background execution
        
        Args:
            func: Function to execute
            task_type: CPU_BOUND or IO_BOUND
            description: Human-readable task description
            user_id: User who initiated the task
            file_id: Associated file ID (optional)
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            task_id: Unique task identifier
        """
        task_id = str(uuid.uuid4())
        
        # Create task info
        task_info = TaskInfo(
            task_id=task_id,
            task_type=task_type,
            description=description,
            user_id=user_id,
            file_id=file_id
        )
        
        # Store task info (thread-safe)
        with self._tasks_lock:
            self.tasks[task_id] = task_info
        
        # Select appropriate executor
        executor = (
            self.cpu_executor if task_type == TaskType.CPU_BOUND
            else self.io_executor
        )
        
        # Wrap function to update task status and pass progress callback
        def wrapped_func():
            try:
                task_info.status = TaskStatus.RUNNING
                task_info.started_at = datetime.utcnow()
                logger.info(f"🚀 Task {task_id} started: {description}")
                
                # Create progress callback
                def progress_callback(
                    progress: float,
                    stage: ProcessingStage = None,
                    details: str = None,
                    **kwargs
                ):
                    """Progress callback that tasks can use to report progress"""
                    task_info.update_progress(progress, stage, details, **kwargs)
                
                # Inject progress callback into kwargs
                kwargs['progress_callback'] = progress_callback
                kwargs['task_id'] = task_id
                kwargs['user_id'] = user_id
                if file_id:
                    kwargs['file_id'] = file_id
                
                # Execute the actual function
                result = func(*args, **kwargs)
                
                task_info.status = TaskStatus.COMPLETED
                task_info.result = result
                task_info.completed_at = datetime.utcnow()
                task_info.progress = 100.0
                task_info.current_stage = ProcessingStage.COMPLETE
                
                logger.info(
                    f"✅ Task {task_id} completed in "
                    f"{task_info._get_processing_time():.2f}s"
                )
                
                return result
                
            except Exception as e:
                task_info.status = TaskStatus.FAILED
                task_info.error = str(e)
                task_info.completed_at = datetime.utcnow()
                task_info.current_stage = ProcessingStage.FAILED
                
                logger.error(f"❌ Task {task_id} failed: {str(e)}", exc_info=True)
                raise
        
        # Submit to executor
        future = executor.submit(wrapped_func)
        task_info.future = future
        
        logger.info(
            f"📤 Task {task_id} submitted to {task_type.value} pool: {description}"
        )
        
        return task_id
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a task (thread-safe)"""
        with self._tasks_lock:
            task_info = self.tasks.get(task_id)
            if not task_info:
                return None
            return task_info.to_dict()
    
    def get_task_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """Get task result (blocking until complete)"""
        with self._tasks_lock:
            task_info = self.tasks.get(task_id)
        
        if not task_info:
            raise ValueError(f"Task {task_id} not found")
        
        if not task_info.future:
            raise ValueError(f"Task {task_id} has no associated future")
        
        # This will block until result is ready or timeout
        return task_info.future.result(timeout=timeout)
    
    def cancel_task(self, task_id: str) -> bool:
        """Attempt to cancel a task"""
        with self._tasks_lock:
            task_info = self.tasks.get(task_id)
        
        if not task_info or not task_info.future:
            return False
        
        if task_info.future.cancel():
            task_info.status = TaskStatus.CANCELLED
            task_info.completed_at = datetime.utcnow()
            logger.info(f"🚫 Task {task_id} cancelled")
            return True
        
        return False
    
    def get_user_tasks(self, user_id: str, limit: int = 50) -> list[Dict[str, Any]]:
        """Get all tasks for a specific user (thread-safe)"""
        with self._tasks_lock:
            user_tasks = [
                task.to_dict()
                for task in self.tasks.values()
                if task.user_id == user_id
            ]
        
        # Sort by created_at (most recent first)
        user_tasks.sort(key=lambda x: x['created_at'], reverse=True)
        
        return user_tasks[:limit]
    
    def cleanup_completed_tasks(self, max_age_hours: int = 24):
        """Remove old completed/failed tasks from memory"""
        cutoff_time = datetime.utcnow().timestamp() - (max_age_hours * 3600)
        
        with self._tasks_lock:
            tasks_to_remove = []
            for task_id, task_info in self.tasks.items():
                if task_info.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                    if task_info.completed_at and task_info.completed_at.timestamp() < cutoff_time:
                        tasks_to_remove.append(task_id)
            
            for task_id in tasks_to_remove:
                del self.tasks[task_id]
            
            if tasks_to_remove:
                logger.info(f"🧹 Cleaned up {len(tasks_to_remove)} old tasks")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get task manager statistics (thread-safe)"""
        with self._tasks_lock:
            total_tasks = len(self.tasks)
            
            status_counts = {
                TaskStatus.PENDING: 0,
                TaskStatus.RUNNING: 0,
                TaskStatus.COMPLETED: 0,
                TaskStatus.FAILED: 0,
                TaskStatus.CANCELLED: 0
            }
            
            type_counts = {
                TaskType.CPU_BOUND: 0,
                TaskType.IO_BOUND: 0
            }
            
            for task in self.tasks.values():
                status_counts[task.status] += 1
                type_counts[task.task_type] += 1
        
        return {
            "total_tasks": total_tasks,
            "by_status": {k.value: v for k, v in status_counts.items()},
            "by_type": {k.value: v for k, v in type_counts.items()},
            "io_workers": self.io_executor._max_workers,
            "cpu_workers": self.cpu_executor._max_workers
        }
    
    def shutdown(self, wait: bool = True):
        """Shutdown the task manager and all executors"""
        logger.info("🛑 Shutting down task manager...")
        self.io_executor.shutdown(wait=wait)
        self.cpu_executor.shutdown(wait=wait)
        logger.info("✅ Task manager shutdown complete")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown(wait=True)


# Global task manager instance
_task_manager: Optional[TaskManager] = None
_manager_lock = threading.Lock()


def get_task_manager() -> TaskManager:
    """Get or create the global task manager instance (thread-safe)"""
    global _task_manager
    
    if _task_manager is None:
        with _manager_lock:
            # Double-check locking pattern
            if _task_manager is None:
                _task_manager = TaskManager(
                    max_io_workers=20,
                    max_cpu_workers=4
                )
    
    return _task_manager


def shutdown_task_manager():
    """Shutdown the global task manager"""
    global _task_manager
    
    if _task_manager is not None:
        with _manager_lock:
            if _task_manager is not None:
                _task_manager.shutdown(wait=True)
                _task_manager = None