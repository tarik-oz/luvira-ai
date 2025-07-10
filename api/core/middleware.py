"""
Middleware for Hair Segmentation API
"""

import time
import logging
from typing import Callable
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
        
        # Log request
        logger.info(
            f"Request: {request.method} {request.url.path} - "
            f"Client: {request.client.host if request.client else 'unknown'}"
        )
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Log response
            logger.info(
                f"Response: {response.status_code} - "
                f"Duration: {duration:.3f}s - "
                f"Path: {request.url.path}"
            )
            
            return response
            
        except Exception as e:
            # Calculate duration
            duration = time.time() - start_time
            
            # Log error
            logger.error(
                f"Error: {str(e)} - "
                f"Duration: {duration:.3f}s - "
                f"Path: {request.url.path}"
            )
            raise


class CORSMiddleware(BaseHTTPMiddleware):
    """Custom CORS middleware with configurable origins"""
    
    def __init__(self, app: ASGIApp, origins: list = None, methods: list = None, headers: list = None):
        super().__init__(app)
        self.origins = origins or ["*"]
        self.methods = methods or ["*"]
        self.headers = headers or ["*"]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Handle preflight requests
        if request.method == "OPTIONS":
            response = Response()
        else:
            response = await call_next(request)
        
        # Add CORS headers to response
        origin = request.headers.get("origin")
        
        # Origin header
        if "*" in self.origins or (origin and origin in self.origins):
            response.headers["Access-Control-Allow-Origin"] = "*" if "*" in self.origins else origin
        
        # Methods header
        if "*" in self.methods:
            response.headers["Access-Control-Allow-Methods"] = "*"
        else:
            response.headers["Access-Control-Allow-Methods"] = ", ".join(self.methods)
        
        # Headers header
        if "*" in self.headers:
            response.headers["Access-Control-Allow-Headers"] = "*"
        else:
            response.headers["Access-Control-Allow-Headers"] = ", ".join(self.headers)
        
        # Credentials header
        response.headers["Access-Control-Allow-Credentials"] = "true"
        
        return response 