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


@router.get("/{user_id}/{kb_id}/{file_id}.md")
async def get_markdown_file(user_id: str, kb_id: str, file_id: str):
    """
    Retrieve the processed markdown file by user_id, kb_id, and file_id
    
    Args:
        user_id: The user identifier
        kb_id: The knowledge base identifier
        file_id: The unique file identifier (without .md extension in URL path)
        
    Returns:
        Plain text markdown content with appropriate headers
        
    Example:
        GET /api/markdown_output/123/6f8f7357-f809-42f7-a2d0-81fb71ab53dd/6cdd7148-4a14-4534-8123-123b9d07c4d8.md
    """
    try:
        # Construct markdown file path with user isolation
        markdown_dir = Path(settings.markdown_dir) / user_id / kb_id
        markdown_file = markdown_dir / f"{file_id}.md"
        
        logger.debug(f"Looking for markdown file: {markdown_file}")
        
        # Check if file exists
        if not markdown_file.exists():
            logger.warning(f"Markdown file not found: {markdown_file}")
            raise HTTPException(
                status_code=404,
                detail=f"Markdown file not found for file_id: {file_id} in KB: {kb_id}"
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
        
        logger.info(f"✓ Served markdown file: {file_id} for user {user_id}")
        
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


@router.get("/{user_id}/{kb_id}/{file_id}.md/download")
async def download_markdown_file(user_id: str, kb_id: str, file_id: str):
    """
    Download the processed markdown file by user_id, kb_id, and file_id
    
    Same as GET but forces download instead of inline display
    
    Args:
        user_id: The user identifier
        kb_id: The knowledge base identifier
        file_id: The unique file identifier
        
    Returns:
        Markdown file as downloadable attachment
        
    Example:
        GET /api/markdown_output/123/6f8f7357-f809-42f7-a2d0-81fb71ab53dd/6cdd7148-4a14-4534-8123-123b9d07c4d8.md/download
    """
    try:
        markdown_dir = Path(settings.markdown_dir) / user_id / kb_id
        markdown_file = markdown_dir / f"{file_id}.md"
        
        logger.debug(f"Looking for markdown file to download: {markdown_file}")
        
        if not markdown_file.exists():
            logger.warning(f"Markdown file not found: {markdown_file}")
            raise HTTPException(
                status_code=404,
                detail=f"Markdown file not found for file_id: {file_id} in KB: {kb_id}"
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
        
        logger.info(f"✓ Downloaded markdown file: {file_id} for user {user_id}")
        
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


@router.get("/list/{user_id}/{kb_id}")
async def list_markdown_files(user_id: str, kb_id: str):
    """
    List all available markdown files for a specific user and knowledge base
    
    Args:
        user_id: The user identifier
        kb_id: The knowledge base identifier (optional path, for organized listing)
        
    Returns:
        List of markdown files with metadata for the user/KB
    """
    try:
        markdown_dir = Path(settings.markdown_dir) / user_id / kb_id
        
        if not markdown_dir.exists():
            logger.debug(f"Markdown directory not found: {markdown_dir}")
            return {
                "total_files": 0,
                "user_id": user_id,
                "kb_id": kb_id,
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
                    "url": f"/api/markdown_output/{user_id}/{kb_id}/{file_id}.md",
                    "download_url": f"/api/markdown_output/{user_id}/{kb_id}/{file_id}.md/download"
                })
        
        # Sort by modified time (most recent first)
        files.sort(key=lambda x: x['modified_time'], reverse=True)
        
        logger.info(f"✓ Listed {len(files)} markdown files for user {user_id}, KB {kb_id}")
        
        return {
            "total_files": len(files),
            "user_id": user_id,
            "kb_id": kb_id,
            "files": files
        }
    
    except Exception as e:
        logger.error(f"Error listing markdown files for user {user_id}, KB {kb_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error listing markdown files: {str(e)}"
        )


@router.delete("/{user_id}/{kb_id}/{file_id}.md")
async def delete_markdown_file(user_id: str, kb_id: str, file_id: str):
    """
    Delete a markdown file by user_id, kb_id, and file_id
    
    Args:
        user_id: The user identifier
        kb_id: The knowledge base identifier
        file_id: The unique file identifier
        
    Returns:
        Success message
    """
    try:
        markdown_dir = Path(settings.markdown_dir) / user_id / kb_id
        markdown_file = markdown_dir / f"{file_id}.md"
        
        logger.debug(f"Attempting to delete markdown file: {markdown_file}")
        
        if not markdown_file.exists():
            logger.warning(f"Markdown file not found: {markdown_file}")
            raise HTTPException(
                status_code=404,
                detail=f"Markdown file not found for file_id: {file_id} in KB: {kb_id}"
            )
        
        try:
            markdown_file.unlink()
            logger.info(f"✓ Deleted markdown file: {file_id}.md for user {user_id}")
            
            return {
                "message": f"Markdown file '{file_id}.md' deleted successfully",
                "user_id": user_id,
                "kb_id": kb_id,
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