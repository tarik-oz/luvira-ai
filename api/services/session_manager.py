"""
Session management service for caching images and masks
"""

import json
import logging
import time
import uuid
import threading
import shutil
import atexit
from pathlib import Path
from typing import Dict
import numpy as np

from ..config import SESSION_CONFIG
from ..core.exceptions import SessionExpiredException

logger = logging.getLogger(__name__)

# Cache configuration from settings
CACHE_DIR = SESSION_CONFIG["cache_dir"]
CACHE_DIR.mkdir(exist_ok=True)
CACHE_TIMEOUT = SESSION_CONFIG["session_timeout_minutes"] * 60  # Convert to seconds
CLEANUP_INTERVAL = SESSION_CONFIG["cleanup_interval_minutes"] * 60  # Convert to seconds


class SessionManager:
    """Manages image and mask caching sessions"""
    
    def __init__(self):
        self._cleanup_timer = None
        if SESSION_CONFIG.get("auto_cleanup_on_startup", True):
            self.cleanup_expired_sessions()
        self._start_cleanup_timer()
        # Cleanup on exit
        if SESSION_CONFIG.get("auto_cleanup_on_shutdown", True):
            atexit.register(self.cleanup_all_sessions)
    
    def _get_session_metadata(self, session_dir: Path) -> dict:
        """Get session metadata from file"""
        metadata_file = session_dir / "metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file) as f:
                    return json.load(f)
            except Exception:
                pass
        return {}
    
    def _is_session_expired(self, metadata: dict) -> bool:
        """Check if session is expired based on metadata"""
        timestamp = metadata.get('timestamp', 0)
        return time.time() - timestamp > CACHE_TIMEOUT
    
    def create_session(self) -> str:
        """
        Create a new session ID
        
        Returns:
            Session ID
        """
        session_id = f"session_{uuid.uuid4().hex[:12]}"
        logger.debug(f"Created new session: {session_id}")
        return session_id
    
    def save_session_data(self, session_id: str, image: np.ndarray, mask: np.ndarray) -> None:
        """
        Save session data to temp files
        
        Args:
            session_id: Session identifier
            image: Original image array
            mask: Hair mask array
        """
        try:
            session_dir = CACHE_DIR / session_id
            session_dir.mkdir(exist_ok=True)
            
            # Save numpy arrays
            np.save(session_dir / "image.npy", image)
            np.save(session_dir / "mask.npy", mask)
            
            # Save metadata
            metadata = {
                'timestamp': time.time(),
                'session_id': session_id,
                'image_shape': image.shape,
                'mask_shape': mask.shape
            }
            
            with open(session_dir / "metadata.json", 'w') as f:
                json.dump(metadata, f)
                
            logger.info(f"Session data saved: {session_id} (Image: {image.shape}, Mask: {mask.shape})")
            
        except Exception as e:
            logger.error(f"Failed to save session {session_id}: {str(e)}")
            raise Exception(f"Failed to save session data: {str(e)}")
    
    def load_session_data(self, session_id: str) -> Dict[str, np.ndarray]:
        """
        Load session data from temp files
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dictionary with image and mask arrays
            
        Raises:
            SessionExpiredException: If session doesn't exist or is expired
        """
        try:
            # Check session validity (will raise SessionExpiredException if invalid)
            self.check_session_validity(session_id)
            
            # Session is valid, load arrays
            session_dir = CACHE_DIR / session_id
            image = np.load(session_dir / "image.npy")
            mask = np.load(session_dir / "mask.npy")
            
            logger.debug(f"Session data loaded: {session_id}")
            return {'image': image, 'mask': mask}
            
        except SessionExpiredException:
            # Re-raise session exceptions as-is
            raise
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {str(e)}")
            raise Exception(f"Failed to load session data: {str(e)}")
    
    def check_session_validity(self, session_id: str) -> None:
        """
        Check if session exists and is valid, raise exception if not
        
        Args:
            session_id: Session identifier
            
        Raises:
            SessionExpiredException: If session doesn't exist or is expired
        """
        try:
            session_dir = CACHE_DIR / session_id
            if not session_dir.exists():
                raise SessionExpiredException(session_id, "Session not found")
            
            metadata = self._get_session_metadata(session_dir)
            if metadata and self._is_session_expired(metadata):
                # Clean up expired session
                self.cleanup_session(session_id)
                elapsed_minutes = int((time.time() - metadata.get('timestamp', 0)) / 60)
                raise SessionExpiredException(
                    session_id, 
                    f"Session expired (created {elapsed_minutes} minutes ago)"
                )
            
            # Session is valid, no exception raised
            
        except SessionExpiredException:
            # Re-raise session exceptions as-is
            raise
        except Exception as e:
            logger.error(f"Error checking session {session_id}: {str(e)}")
            raise SessionExpiredException(session_id, f"Session validation failed: {str(e)}")
    
    def cleanup_session(self, session_id: str) -> None:
        """
        Clean up a specific session
        
        Args:
            session_id: Session identifier
        """
        try:
            session_dir = CACHE_DIR / session_id
            if session_dir.exists():
                shutil.rmtree(session_dir)
                logger.info(f"Session cleaned up: {session_id}")
        except Exception as e:
            logger.warning(f"Failed to cleanup session {session_id}: {str(e)}")
    
    def cleanup_expired_sessions(self) -> int:
        """
        Clean up all expired sessions
        
        Returns:
            Number of sessions cleaned up
        """
        cleaned_count = 0
        try:
            current_time = time.time()
            
            for session_dir in CACHE_DIR.iterdir():
                if not session_dir.is_dir():
                    continue
                    
                try:
                    metadata = self._get_session_metadata(session_dir)
                    if not metadata or self._is_session_expired(metadata):
                        self.cleanup_session(session_dir.name)
                        cleaned_count += 1
                        
                except Exception as e:
                    logger.warning(f"Error checking session {session_dir.name}: {str(e)}")
                    
            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} expired sessions")
                
        except Exception as e:
            logger.error(f"Failed to cleanup expired sessions: {str(e)}")
            
        return cleaned_count
    
    def cleanup_all_sessions(self) -> int:
        """
        Clean up all sessions on API shutdown
        
        Returns:
            Number of sessions cleaned up
        """
        cleaned_count = 0
        try:
            if CACHE_DIR.exists():
                for session_dir in CACHE_DIR.iterdir():
                    if session_dir.is_dir() and session_dir.name.startswith("session_"):
                        shutil.rmtree(session_dir)
                        cleaned_count += 1
                        logger.debug(f"API shutdown - cleaned session: {session_dir.name}")
                
                logger.info(f"API SHUTDOWN: {cleaned_count} sessions cleaned up")
            else:
                logger.debug("No session directory to clean up")
                
        except Exception as e:
            logger.error(f"Failed to cleanup all sessions on shutdown: {str(e)}")
            
        return cleaned_count
    
    def get_cache_stats(self) -> Dict:
        """
        Get cache statistics
        
        Returns:
            Dictionary with cache stats
        """
        try:
            stats = {
                'total_sessions': 0,
                'active_sessions': 0,
                'expired_sessions': 0,
                'total_size_mb': 0
            }
            
            current_time = time.time()
            
            for session_dir in CACHE_DIR.iterdir():
                if not session_dir.is_dir():
                    continue
                    
                stats['total_sessions'] += 1
                
                # Calculate directory size
                size = sum(f.stat().st_size for f in session_dir.glob('**/*') if f.is_file())
                stats['total_size_mb'] += size / (1024 * 1024)
                
                # Check if expired
                metadata = self._get_session_metadata(session_dir)
                if metadata and not self._is_session_expired(metadata):
                    stats['active_sessions'] += 1
                else:
                    stats['expired_sessions'] += 1
            
            stats['total_size_mb'] = round(stats['total_size_mb'], 2)
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get cache stats: {str(e)}")
            return {'error': str(e)}
    
    def _start_cleanup_timer(self):
        """
        Start a background thread to periodically clean up expired sessions
        It checks every 10 minutes
        """
        def cleanup_worker():
            interval_minutes = SESSION_CONFIG["cleanup_interval_minutes"]
            logger.info(f"Session cleanup timer started - checking every {interval_minutes} minutes")
            while True:
                try:
                    time.sleep(CLEANUP_INTERVAL)
                    logger.debug("Running periodic session cleanup check...")
                    cleaned = self.cleanup_expired_sessions()
                    if cleaned > 0:
                        logger.info(f"Periodic cleanup: {cleaned} expired sessions removed")
                except Exception as e:
                    logger.error(f"Cleanup timer error: {str(e)}")
        
        self._cleanup_timer = threading.Thread(target=cleanup_worker, daemon=True)
        self._cleanup_timer.start()
        logger.info(f"Session cleanup timer STARTED (checks every {SESSION_CONFIG['cleanup_interval_minutes']} minutes)")
    
    def __del__(self):
        """Cleanup on destruction"""
        if self._cleanup_timer and self._cleanup_timer.is_alive():
            logger.debug("Session manager cleanup timer stopped")


# Global session manager instance
session_manager = SessionManager()
