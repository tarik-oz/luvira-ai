"""
Session management service for caching images and masks
"""

import tempfile
import logging
import time
import uuid
import threading
import shutil
import atexit
from pathlib import Path
from typing import Dict, Optional
import numpy as np

logger = logging.getLogger(__name__)

# Cache configuration - Store in project directory
PROJECT_ROOT = Path(__file__).parent.parent.parent  # Go up to project root
CACHE_DIR = PROJECT_ROOT / "session_data"
CACHE_DIR.mkdir(exist_ok=True)
CACHE_TIMEOUT = 30 * 60  # 30 minutes


class SessionManager:
    """Manages image and mask caching sessions"""
    
    def __init__(self):
        self._cleanup_timer = None
        self._start_cleanup_timer()
        # Cleanup on exit
        atexit.register(self.cleanup_all_sessions)
    
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
            
            import json
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
        """
        try:
            session_dir = CACHE_DIR / session_id
            if not session_dir.exists():
                raise Exception(f"Session {session_id} not found")
            
            # Check if session expired
            metadata_file = session_dir / "metadata.json"
            if metadata_file.exists():
                import json
                metadata = json.load(open(metadata_file))
                timestamp = metadata.get('timestamp', 0)
                
                if time.time() - timestamp > CACHE_TIMEOUT:
                    # Clean up expired session
                    self.cleanup_session(session_id)
                    raise Exception(f"Session {session_id} expired (created {int((time.time() - timestamp) / 60)} minutes ago)")
            
            # Load arrays
            image = np.load(session_dir / "image.npy")
            mask = np.load(session_dir / "mask.npy")
            
            logger.debug(f"Session data loaded: {session_id}")
            return {'image': image, 'mask': mask}
            
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {str(e)}")
            raise Exception(f"Failed to load session data: {str(e)}")
    
    def session_exists(self, session_id: str) -> bool:
        """
        Check if session exists and is valid
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if session exists and not expired
        """
        try:
            session_dir = CACHE_DIR / session_id
            if not session_dir.exists():
                return False
            
            metadata_file = session_dir / "metadata.json"
            if metadata_file.exists():
                import json
                metadata = json.load(open(metadata_file))
                timestamp = metadata.get('timestamp', 0)
                
                if time.time() - timestamp > CACHE_TIMEOUT:
                    # Session expired
                    return False
            
            return True
            
        except Exception:
            return False
    
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
                    metadata_file = session_dir / "metadata.json"
                    if metadata_file.exists():
                        import json
                        metadata = json.load(open(metadata_file))
                        timestamp = metadata.get('timestamp', 0)
                        
                        if current_time - timestamp > CACHE_TIMEOUT:
                            self.cleanup_session(session_dir.name)
                            cleaned_count += 1
                    else:
                        # No metadata, probably old format or corrupted
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
                metadata_file = session_dir / "metadata.json"
                if metadata_file.exists():
                    try:
                        import json
                        metadata = json.load(open(metadata_file))
                        timestamp = metadata.get('timestamp', 0)
                        
                        if current_time - timestamp > CACHE_TIMEOUT:
                            stats['expired_sessions'] += 1
                        else:
                            stats['active_sessions'] += 1
                    except:
                        stats['expired_sessions'] += 1
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
            logger.info("Session cleanup timer started - checking every 10 minutes")
            while True:
                try:
                    time.sleep(10 * 60)  # Wait for 10 minutes
                    logger.debug("Running periodic session cleanup check...")
                    cleaned = self.cleanup_expired_sessions()
                    if cleaned > 0:
                        logger.info(f"Periodic cleanup: {cleaned} expired sessions removed")
                except Exception as e:
                    logger.error(f"Cleanup timer error: {str(e)}")
        
        self._cleanup_timer = threading.Thread(target=cleanup_worker, daemon=True)
        self._cleanup_timer.start()
        logger.info("Session cleanup timer STARTED (checks every 10 minutes for expired sessions)")
    
    def __del__(self):
        """Cleanup on destruction"""
        if self._cleanup_timer and self._cleanup_timer.is_alive():
            logger.debug("Session manager cleanup timer stopped")


# Global session manager instance
session_manager = SessionManager()
