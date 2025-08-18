"""
Middleware for Hair Segmentation API
"""

import time
import logging
import json
from typing import Callable
import uuid
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging requests and responses"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Start time
        start_time = time.time()
        
        # Correlate logs with a request id
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        # Log request (structured)
        logger.info(json.dumps({
            "event": "request",
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "client_ip": request.client.host if request.client else "unknown",
        }))
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Log response (structured)
            logger.info(json.dumps({
                "event": "response",
                "request_id": request_id,
                "status": int(response.status_code),
                "duration_ms": int(duration * 1000),
                "path": request.url.path,
            }))
            
            return response
            
        except Exception as e:
            # Calculate duration
            duration = time.time() - start_time
            
            # Log error (structured)
            logger.error(json.dumps({
                "event": "error",
                "request_id": request_id,
                "error": str(e),
                "duration_ms": int(duration * 1000),
                "path": request.url.path,
            }))
            raise 