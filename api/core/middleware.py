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
            response_data = {
                "event": "response",
                "request_id": request_id,
                "status": int(response.status_code),
                "duration_ms": int(duration * 1000),
                "path": request.url.path,
            }
            
            # Add error details for 4xx and 5xx responses
            if response.status_code >= 400:
                try:
                    # Try to get response body for error details
                    response_body = b""
                    async for chunk in response.body_iterator:
                        response_body += chunk
                    
                    # Parse response body to get error code
                    if response_body:
                        try:
                            error_response = json.loads(response_body.decode())
                            if "error_code" in error_response:
                                response_data["error_code"] = error_response["error_code"]
                        except json.JSONDecodeError:
                            pass
                    
                    # Create new response with the same body
                    response = Response(
                        content=response_body,
                        status_code=response.status_code,
                        headers=dict(response.headers),
                        media_type=response.media_type
                    )
                except Exception:
                    # If we can't read response body, continue without error details
                    pass
            
            logger.info(json.dumps(response_data))
            
            return response
            
        except Exception as e:
            # Calculate duration
            duration = time.time() - start_time
            
            # Log error (structured)
            error_data = {
                "event": "error",
                "request_id": request_id,
                "error": str(e),
                "duration_ms": int(duration * 1000),
                "path": request.url.path,
            }
            
            # Add custom exception details if available
            if hasattr(e, 'error_code'):
                error_data["error_code"] = e.error_code
            else:
                # Default error code for unknown exceptions
                error_data["error_code"] = "UNKNOWN_ERROR"
                
            if hasattr(e, 'extra_data'):
                error_data["extra_data"] = e.extra_data
            if hasattr(e, 'detail'):
                error_data["detail"] = e.detail
                
            logger.error(json.dumps(error_data))
            raise 