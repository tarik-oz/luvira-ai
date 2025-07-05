#!/usr/bin/env python3
"""
Script to run the FastAPI server
"""

import uvicorn
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from model.config import API_CONFIG

def print_startup_info():
    """Print startup information"""
    print("Starting Hair Segmentation API...")
    print(f"Host: {API_CONFIG['host']}")
    print(f"Port: {API_CONFIG['port']}")
    print(f"Reload: {API_CONFIG['reload']}")
    print(f"Log Level: {API_CONFIG['log_level']}")
    print()
    print("API Documentation:")
    print(f"   • Swagger UI: http://localhost:{API_CONFIG['port']}/docs")
    print(f"   • ReDoc: http://localhost:{API_CONFIG['port']}/redoc")
    print()
    print("Health Check:")
    print(f"   • Status: http://localhost:{API_CONFIG['port']}/")
    print(f"   • Health: http://localhost:{API_CONFIG['port']}/health")
    print()
    print("Model Endpoints:")
    print(f"   • Model Info: http://localhost:{API_CONFIG['port']}/model-info")
    print(f"   • Predict Mask: http://localhost:{API_CONFIG['port']}/predict-mask")
    print()

if __name__ == "__main__":
    try:
        print_startup_info()
        
        uvicorn.run(
            "api.main:app",
            host=API_CONFIG["host"],
            port=API_CONFIG["port"],
            reload=API_CONFIG["reload"],
            log_level=API_CONFIG["log_level"]
        )
    except KeyboardInterrupt:
        print("\nAPI server stopped by user")
    except Exception as e:
        print(f"Error starting API server: {e}")
        sys.exit(1) 