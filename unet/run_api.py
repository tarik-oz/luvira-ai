#!/usr/bin/env python3
"""
Script to run the FastAPI server
"""

import uvicorn
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

if __name__ == "__main__":
    print("Starting Hair Segmentation API...")
    print("API Documentation: http://localhost:8000/docs")
    print("Alternative docs: http://localhost:8000/redoc")
    print("Health check: http://localhost:8000/")
    print()
    
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 