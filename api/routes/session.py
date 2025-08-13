"""
Session management routes for Hair Segmentation API
"""

from fastapi import APIRouter, HTTPException
from ..services.session_manager import session_manager

router = APIRouter()


@router.get("/session-stats")
async def get_session_stats():
    """
    Get current session statistics
    
    Returns:
        Dictionary with cache statistics including:
        - total_sessions: Total number of sessions
        - active_sessions: Number of active sessions
        - expired_sessions: Number of expired sessions
        - total_size_mb: Total cache size in MB
    """
    return session_manager.get_cache_stats()


@router.delete("/cleanup-session/{session_id}")
async def cleanup_session(session_id: str):
    """
    Clean up specific session data
    
    Args:
        session_id: Session identifier to cleanup
        
    Returns:
        Success message
    """
    try:
        session_manager.cleanup_session(session_id)
        return {"message": f"Session {session_id} cleaned up successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Session cleanup failed: {str(e)}"
        )


@router.post("/cleanup-all-sessions")
async def cleanup_all_sessions():
    """
    Manual cleanup of ALL sessions
    
    This is useful for testing or when you need to clear all cached data.
    
    Returns:
        Number of cleaned sessions
    """
    cleaned = session_manager.cleanup_all_sessions()
    return {"message": f"Cleaned up {cleaned} sessions", "cleaned_count": cleaned}
