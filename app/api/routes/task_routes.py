"""
Task Status API Routes
Allows users to check the status of background tasks
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Optional
import logging

from app.core.dependencies import get_user_id
from app.core.task_manager import get_task_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/tasks", tags=["Task Status"])


@router.get("/{task_id}")
async def get_task_status(
    task_id: str,
    user_id: str = Depends(get_user_id)
):
    """
    Get the status of a background task
    
    Headers:
        X-User-ID: User identifier (REQUIRED)
    
    Args:
        task_id: Task identifier returned from processing endpoint
        
    Returns:
        Task status including:
        - task_id: Task identifier
        - status: pending, running, completed, failed, cancelled
        - progress: Progress percentage (0-100)
        - created_at: When task was created
        - started_at: When task started running
        - completed_at: When task finished
        - processing_time: Total processing time in seconds
        - error: Error message if failed
        
    Example Response (Completed):
        {
            "task_id": "abc-123",
            "task_type": "cpu_bound",
            "description": "Process document: example.pdf",
            "user_id": "user_123",
            "file_id": "file_456",
            "status": "completed",
            "progress": 100.0,
            "created_at": "2024-01-30T10:00:00",
            "started_at": "2024-01-30T10:00:01",
            "completed_at": "2024-01-30T10:02:30",
            "processing_time": 149.5,
            "error": null
        }
        
    Example Response (Failed):
        {
            "task_id": "abc-123",
            "status": "failed",
            "error": "Extraction failed: Unsupported file format",
            ...
        }
    """
    task_manager = get_task_manager()
    
    task_status = task_manager.get_task_status(task_id)
    
    if not task_status:
        raise HTTPException(
            status_code=404,
            detail=f"Task not found: {task_id}"
        )
    
    # Verify user owns this task
    if task_status['user_id'] != user_id:
        raise HTTPException(
            status_code=403,
            detail="Access denied: You don't have permission to view this task"
        )
    
    return task_status


@router.get("/file/{file_id}")
async def get_file_task_status(
    file_id: str,
    user_id: str = Depends(get_user_id)
):
    """
    Get the most recent task status for a specific file
    
    Headers:
        X-User-ID: User identifier (REQUIRED)
    
    Args:
        file_id: File identifier
        
    Returns:
        Task status or 404 if no task found for this file
        
    This is useful when you know the file_id but not the task_id
    """
    task_manager = get_task_manager()
    
    # Get all user tasks
    user_tasks = task_manager.get_user_tasks(user_id, limit=100)
    
    # Find tasks for this file (most recent first)
    file_tasks = [
        task for task in user_tasks
        if task.get('file_id') == file_id
    ]
    
    if not file_tasks:
        raise HTTPException(
            status_code=404,
            detail=f"No tasks found for file: {file_id}"
        )
    
    # Return most recent task
    return file_tasks[0]


@router.get("/user/all")
async def get_user_tasks(
    user_id: str = Depends(get_user_id),
    limit: int = 50,
    status_filter: Optional[str] = None
):
    """
    Get all tasks for the current user
    
    Headers:
        X-User-ID: User identifier (REQUIRED)
    
    Query Parameters:
        limit: Maximum number of tasks to return (default: 50, max: 200)
        status_filter: Filter by status (pending, running, completed, failed, cancelled)
        
    Returns:
        List of task statuses sorted by creation time (most recent first)
        
    Example:
        GET /api/tasks/user/all?limit=20&status_filter=completed
    """
    # Validate limit
    limit = min(max(1, limit), 200)
    
    task_manager = get_task_manager()
    
    # Get user tasks
    tasks = task_manager.get_user_tasks(user_id, limit=limit)
    
    # Apply status filter if provided
    if status_filter:
        valid_statuses = ['pending', 'running', 'completed', 'failed', 'cancelled']
        if status_filter not in valid_statuses:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status filter. Must be one of: {', '.join(valid_statuses)}"
            )
        tasks = [task for task in tasks if task['status'] == status_filter]
    
    return {
        "total_tasks": len(tasks),
        "limit": limit,
        "status_filter": status_filter,
        "tasks": tasks
    }


@router.delete("/{task_id}")
async def cancel_task(
    task_id: str,
    user_id: str = Depends(get_user_id)
):
    """
    Attempt to cancel a running or pending task
    
    Headers:
        X-User-ID: User identifier (REQUIRED)
    
    Args:
        task_id: Task identifier to cancel
        
    Returns:
        Success message if cancelled, error if unable to cancel
        
    Note:
        - Only pending or recently running tasks can be cancelled
        - Tasks that are already processing may not be cancelled immediately
        - Completed or failed tasks cannot be cancelled
    """
    task_manager = get_task_manager()
    
    # Get task status
    task_status = task_manager.get_task_status(task_id)
    
    if not task_status:
        raise HTTPException(
            status_code=404,
            detail=f"Task not found: {task_id}"
        )
    
    # Verify user owns this task
    if task_status['user_id'] != user_id:
        raise HTTPException(
            status_code=403,
            detail="Access denied: You don't have permission to cancel this task"
        )
    
    # Check if task can be cancelled
    if task_status['status'] in ['completed', 'failed', 'cancelled']:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel task with status: {task_status['status']}"
        )
    
    # Attempt to cancel
    success = task_manager.cancel_task(task_id)
    
    if success:
        return {
            "message": f"Task {task_id} cancelled successfully",
            "task_id": task_id,
            "previous_status": task_status['status']
        }
    else:
        raise HTTPException(
            status_code=400,
            detail="Task could not be cancelled (may have already started execution)"
        )


@router.get("/stats/summary")
async def get_task_stats(user_id: str = Depends(get_user_id)):
    """
    Get task statistics for the current user
    
    Headers:
        X-User-ID: User identifier (REQUIRED)
        
    Returns:
        Summary statistics about user's tasks
        
    Example Response:
        {
            "user_id": "user_123",
            "total_tasks": 45,
            "by_status": {
                "pending": 2,
                "running": 3,
                "completed": 38,
                "failed": 2,
                "cancelled": 0
            },
            "by_type": {
                "cpu_bound": 25,
                "io_bound": 20
            },
            "recent_tasks": [...] // Last 10 tasks
        }
    """
    task_manager = get_task_manager()
    
    # Get user tasks
    all_tasks = task_manager.get_user_tasks(user_id, limit=1000)
    
    # Calculate statistics
    status_counts = {
        'pending': 0,
        'running': 0,
        'completed': 0,
        'failed': 0,
        'cancelled': 0
    }
    
    type_counts = {
        'cpu_bound': 0,
        'io_bound': 0
    }
    
    for task in all_tasks:
        status_counts[task['status']] += 1
        type_counts[task['task_type']] += 1
    
    return {
        "user_id": user_id,
        "total_tasks": len(all_tasks),
        "by_status": status_counts,
        "by_type": type_counts,
        "recent_tasks": all_tasks[:10]  # Last 10 tasks
    }


@router.get("/system/stats")
async def get_system_task_stats():
    """
    Get system-wide task statistics (admin only in production)
    
    Returns:
        Overall task manager statistics
        
    Note:
        In production, this should be protected with admin authentication
    """
    task_manager = get_task_manager()
    return task_manager.get_stats()