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
from typing import Dict, Optional
import numpy as np

try:
    import boto3  # type: ignore
    from botocore.exceptions import BotoCoreError, ClientError  # type: ignore
except Exception:  # pragma: no cover
    boto3 = None  # type: ignore
    BotoCoreError = ClientError = Exception  # type: ignore

from ..config import SESSION_CONFIG
from ..core.exceptions import SessionExpiredException

logger = logging.getLogger(__name__)

BACKEND = SESSION_CONFIG.get("backend", "filesystem").lower()
CACHE_DIR = SESSION_CONFIG["cache_dir"]
if BACKEND == "filesystem":
    CACHE_DIR.mkdir(exist_ok=True)
CACHE_TIMEOUT = SESSION_CONFIG["session_timeout_minutes"] * 60  # Convert to seconds
CLEANUP_INTERVAL = SESSION_CONFIG["cleanup_interval_minutes"] * 60  # Convert to seconds

S3_BUCKET = SESSION_CONFIG.get("s3_bucket", "")
S3_PREFIX = SESSION_CONFIG.get("s3_prefix", "sessions/")
S3_REGION = SESSION_CONFIG.get("s3_region", "")


class _BaseSessionBackend:
    def save(self, session_id: str, image: np.ndarray, mask: np.ndarray) -> None:
        raise NotImplementedError

    def load(self, session_id: str) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    def delete(self, session_id: str) -> None:
        raise NotImplementedError

    def list_session_dirs(self):
        raise NotImplementedError

    def read_metadata(self, session_id: str) -> dict:
        raise NotImplementedError


class FileSystemSessionBackend(_BaseSessionBackend):
    def save(self, session_id: str, image: np.ndarray, mask: np.ndarray) -> None:
        session_dir = CACHE_DIR / session_id
        session_dir.mkdir(exist_ok=True)
        np.save(session_dir / "image.npy", image)
        np.save(session_dir / "mask.npy", mask)
        metadata = {
            'timestamp': time.time(),
            'session_id': session_id,
            'image_shape': image.shape,
            'mask_shape': mask.shape
        }
        with open(session_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f)

    def load(self, session_id: str) -> Dict[str, np.ndarray]:
        session_dir = CACHE_DIR / session_id
        image = np.load(session_dir / "image.npy")
        mask = np.load(session_dir / "mask.npy")
        return {'image': image, 'mask': mask}

    def delete(self, session_id: str) -> None:
        session_dir = CACHE_DIR / session_id
        if session_dir.exists():
            shutil.rmtree(session_dir)

    def list_session_dirs(self):
        if not CACHE_DIR.exists():
            return []
        return [p for p in CACHE_DIR.iterdir() if p.is_dir()]

    def read_metadata(self, session_id: str) -> dict:
        metadata_file = CACHE_DIR / session_id / "metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file) as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}


class S3SessionBackend(_BaseSessionBackend):
    def __init__(self, bucket: str, prefix: str = "sessions/"):
        if boto3 is None:
            raise RuntimeError("boto3 is required for S3 session backend")
        self.bucket = bucket
        self.prefix = prefix.rstrip('/') + '/'
        self._s3 = boto3.client('s3', region_name=S3_REGION or None)

    def _key(self, session_id: str, name: str) -> str:
        return f"{self.prefix}{session_id}/{name}"

    def save(self, session_id: str, image: np.ndarray, mask: np.ndarray) -> None:
        import io
        # npy bytes
        buf_img = io.BytesIO()
        np.save(buf_img, image)
        buf_img.seek(0)
        buf_msk = io.BytesIO()
        np.save(buf_msk, mask)
        buf_msk.seek(0)
        metadata = {
            'timestamp': time.time(),
            'session_id': session_id,
            'image_shape': tuple(int(x) for x in image.shape),
            'mask_shape': tuple(int(x) for x in mask.shape)
        }
        buf_meta = io.BytesIO(json.dumps(metadata).encode('utf-8'))
        try:
            self._s3.upload_fileobj(buf_img, self.bucket, self._key(session_id, 'image.npy'))
            self._s3.upload_fileobj(buf_msk, self.bucket, self._key(session_id, 'mask.npy'))
            self._s3.upload_fileobj(buf_meta, self.bucket, self._key(session_id, 'metadata.json'))
        except (BotoCoreError, ClientError) as e:
            raise RuntimeError(f"S3 upload failed: {e}")

    def load(self, session_id: str) -> Dict[str, np.ndarray]:
        import io
        try:
            buf_img = io.BytesIO()
            self._s3.download_fileobj(self.bucket, self._key(session_id, 'image.npy'), buf_img)
            buf_img.seek(0)
            image = np.load(buf_img, allow_pickle=False)
            buf_msk = io.BytesIO()
            self._s3.download_fileobj(self.bucket, self._key(session_id, 'mask.npy'), buf_msk)
            buf_msk.seek(0)
            mask = np.load(buf_msk, allow_pickle=False)
            return {'image': image, 'mask': mask}
        except (BotoCoreError, ClientError) as e:
            raise RuntimeError(f"S3 download failed: {e}")

    def delete(self, session_id: str) -> None:
        try:
            keys = [
                {'Key': self._key(session_id, 'image.npy')},
                {'Key': self._key(session_id, 'mask.npy')},
                {'Key': self._key(session_id, 'metadata.json')},
            ]
            self._s3.delete_objects(Bucket=self.bucket, Delete={'Objects': keys})
        except (BotoCoreError, ClientError):
            pass

    def list_session_dirs(self):
        # Returns list of (session_id, metadata_dict or None)
        paginator = self._s3.get_paginator('list_objects_v2')
        try:
            for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix, Delimiter='/'):
                for cp in page.get('CommonPrefixes', []):
                    yield cp['Prefix']
        except (BotoCoreError, ClientError):
            return []

    def read_metadata(self, session_id: str) -> dict:
        import io
        try:
            buf = io.BytesIO()
            self._s3.download_fileobj(self.bucket, self._key(session_id, 'metadata.json'), buf)
            buf.seek(0)
            return json.loads(buf.read().decode('utf-8'))
        except Exception:
            return {}


class SessionManager:
    """Manages image and mask caching sessions"""
    
    def __init__(self):
        self._cleanup_timer = None
        # choose backend
        if BACKEND == 's3':
            if not S3_BUCKET:
                raise RuntimeError("SESSION_BACKEND=s3 but S3_BUCKET is not set")
            self._backend: _BaseSessionBackend = S3SessionBackend(S3_BUCKET, S3_PREFIX)
        else:
            self._backend = FileSystemSessionBackend()

        if SESSION_CONFIG.get("auto_cleanup_on_startup", True):
            self.cleanup_expired_sessions()
        self._start_cleanup_timer()
        # Cleanup on exit
        if SESSION_CONFIG.get("auto_cleanup_on_shutdown", True):
            atexit.register(self.cleanup_all_sessions)
    
    def _get_session_metadata(self, session_id: str, session_dir: Optional[Path] = None) -> dict:
        """Get session metadata from backend"""
        try:
            if isinstance(self._backend, FileSystemSessionBackend) and session_dir is not None:
                metadata_file = session_dir / "metadata.json"
                if metadata_file.exists():
                    with open(metadata_file) as f:
                        return json.load(f)
                return {}
            return self._backend.read_metadata(session_id)
        except Exception:
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
            self._backend.save(session_id, image, mask)
                
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
            
            # Session is valid, load arrays from backend
            data = self._backend.load(session_id)
            image = data['image']
            mask = data['mask']
            
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
            # For filesystem, verify dir; S3 will check metadata existence implicitly
            session_dir = CACHE_DIR / session_id
            if isinstance(self._backend, FileSystemSessionBackend) and not session_dir.exists():
                raise SessionExpiredException(session_id, "Session not found", reason="not_found")
            
            metadata = self._get_session_metadata(session_id, session_dir if session_dir.exists() else None)
            if metadata and self._is_session_expired(metadata):
                # Clean up expired session
                self.cleanup_session(session_id)
                elapsed_minutes = int((time.time() - metadata.get('timestamp', 0)) / 60)
                raise SessionExpiredException(
                    session_id, 
                    f"Session expired (created {elapsed_minutes} minutes ago)",
                    reason="expired",
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
            self._backend.delete(session_id)
            if isinstance(self._backend, FileSystemSessionBackend):
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
            
            if isinstance(self._backend, FileSystemSessionBackend):
                for session_dir in CACHE_DIR.iterdir():
                    if not session_dir.is_dir():
                        continue
                    try:
                        metadata = self._get_session_metadata(session_dir.name, session_dir)
                        if not metadata or self._is_session_expired(metadata):
                            self.cleanup_session(session_dir.name)
                            cleaned_count += 1
                    except Exception as e:
                        logger.warning(f"Error checking session {session_dir.name}: {str(e)}")
            else:
                # S3: list prefixes and check metadata timestamps
                try:
                    import re
                    prefix = S3_PREFIX.rstrip('/') + '/'
                    s3 = boto3.client('s3')
                    paginator = s3.get_paginator('list_objects_v2')
                    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
                        contents = page.get('Contents', [])
                        # group by session folder name
                        session_ids = set()
                        for obj in contents:
                            key = obj.get('Key', '')
                            parts = key[len(prefix):].split('/')
                            if parts and parts[0]:
                                session_ids.add(parts[0])
                        for sid in session_ids:
                            metadata = self._get_session_metadata(sid)
                            if not metadata or self._is_session_expired(metadata):
                                self.cleanup_session(sid)
                                cleaned_count += 1
                except Exception as e:
                    logger.warning(f"S3 cleanup listing failed: {e}")
                    
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
