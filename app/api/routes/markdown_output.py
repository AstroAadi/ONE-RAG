"""
Markdown Output Route
Serves extracted markdown files by file_id
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pathlib import Path
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/markdown_output", tags=["Markdown Output"])


@router.get("/{user_id}/{file_id}.md")
async def get_markdown_file(user_id: str, file_id: str):
    """
    Retrieve the processed markdown file by user_id and file_id
    
    Args:
        user_id: The user identifier
        file_id: The unique file identifier (without .md extension in URL path)
        
    Returns:
        Plain text markdown content with appropriate headers
        
    Example:
        GET /api/markdown_output/user_1767769277267_yq5c8ek4f/670d7692-59c7-4cb9-899b-e9ef4c3a7947.md
    """
    try:
        # Construct markdown file path with user isolation
        markdown_dir = Path(settings.markdown_dir) / user_id
        markdown_file = markdown_dir / f"{file_id}.md"
        
        # Check if file exists
        if not markdown_file.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Markdown file not found for file_id: {file_id}"
            )
        
        # Read markdown content
        try:
            with open(markdown_file, 'r', encoding='utf-8') as f:
                markdown_content = f.read()
        except Exception as read_error:
            logger.error(f"Error reading markdown file {file_id}: {str(read_error)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error reading markdown file: {str(read_error)}"
            )
        
        # Return as plain text with markdown content type
        return Response(
            content=markdown_content,
            media_type="text/markdown; charset=utf-8",
            headers={
                "Content-Disposition": f'inline; filename="{file_id}.md"'
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error serving markdown {file_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error serving markdown file: {str(e)}"
        )


@router.get("/{user_id}/{file_id}.md/download")
async def download_markdown_file(user_id: str, file_id: str):
    """
    Download the processed markdown file by user_id and file_id
    
    Same as GET but forces download instead of inline display
    
    Args:
        user_id: The user identifier
        file_id: The unique file identifier
        
    Returns:
        Markdown file as downloadable attachment
        
    Example:
        GET /api/markdown_output/user_1767769277267_yq5c8ek4f/670d7692-59c7-4cb9-899b-e9ef4c3a7947.md/download
    """
    try:
        markdown_dir = Path(settings.markdown_dir) / user_id
        markdown_file = markdown_dir / f"{file_id}.md"
        
        if not markdown_file.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Markdown file not found for file_id: {file_id}"
            )
        
        try:
            with open(markdown_file, 'r', encoding='utf-8') as f:
                markdown_content = f.read()
        except Exception as read_error:
            logger.error(f"Error reading markdown file {file_id}: {str(read_error)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error reading markdown file: {str(read_error)}"
            )
        
        # Return as downloadable file
        return Response(
            content=markdown_content,
            media_type="text/markdown; charset=utf-8",
            headers={
                "Content-Disposition": f'attachment; filename="{file_id}.md"'
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error downloading markdown {file_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error downloading markdown file: {str(e)}"
        )


@router.get("/list/{user_id}")
async def list_markdown_files(user_id: str):
    """
    List all available markdown files for a specific user
    
    Args:
        user_id: The user identifier
        
    Returns:
        List of markdown files with metadata for the user
    """
    try:
        markdown_dir = Path(settings.markdown_dir) / user_id
        
        if not markdown_dir.exists():
            return {
                "total_files": 0,
                "user_id": user_id,
                "files": []
            }
        
        files = []
        for markdown_file in markdown_dir.glob("*.md"):
            if markdown_file.is_file():
                stat = markdown_file.stat()
                
                # Extract file_id from filename (remove .md extension)
                file_id = markdown_file.stem
                
                files.append({
                    "file_id": file_id,
                    "filename": markdown_file.name,
                    "size_bytes": stat.st_size,
                    "size_kb": round(stat.st_size / 1024, 2),
                    "created_time": stat.st_ctime,
                    "modified_time": stat.st_mtime,
                    "url": f"/api/markdown_output/{user_id}/{file_id}.md",
                    "download_url": f"/api/markdown_output/{user_id}/{file_id}.md/download"
                })
        
        # Sort by modified time (most recent first)
        files.sort(key=lambda x: x['modified_time'], reverse=True)
        
        return {
            "total_files": len(files),
            "user_id": user_id,
            "files": files
        }
    
    except Exception as e:
        logger.error(f"Error listing markdown files for user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error listing markdown files: {str(e)}"
        )


@router.delete("/{user_id}/{file_id}.md")
async def delete_markdown_file(user_id: str, file_id: str):
    """
    Delete a markdown file by user_id and file_id
    
    Args:
        user_id: The user identifier
        file_id: The unique file identifier
        
    Returns:
        Success message
    """
    try:
        markdown_dir = Path(settings.markdown_dir) / user_id
        markdown_file = markdown_dir / f"{file_id}.md"
        
        if not markdown_file.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Markdown file not found for file_id: {file_id}"
            )
        
        try:
            markdown_file.unlink()
            logger.info(f"Deleted markdown file: {file_id}.md")
            
            return {
                "message": f"Markdown file '{file_id}.md' deleted successfully",
                "file_id": file_id
            }
        except Exception as delete_error:
            logger.error(f"Error deleting markdown file {file_id}: {str(delete_error)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error deleting markdown file: {str(delete_error)}"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error deleting markdown {file_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting markdown file: {str(e)}"
        )